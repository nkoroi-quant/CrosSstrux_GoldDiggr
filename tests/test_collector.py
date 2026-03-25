"""
Tests for data_layer.collector module.
"""

import os
import json
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Mock MetaTrader5 before importing collector
import sys
from unittest.mock import MagicMock, patch

# Create mock MT5 module
mock_mt5 = MagicMock()
mock_mt5.TIMEFRAME_M1 = 1
mock_mt5.initialize.return_value = True
mock_mt5.symbol_select.return_value = True
mock_mt5.shutdown.return_value = None

# Add to sys.modules
sys.modules['MetaTrader5'] = mock_mt5

from data_layer import collector


class TestLoadSymbolMap:
    """Tests for load_symbol_map function."""
    
    def test_load_existing_config(self, tmp_path):
        """Test loading an existing config file."""
        config_path = tmp_path / "symbol_map.json"
        test_data = {"BTCUSD": "BTCUSD", "XAUUSD": "GOLD"}
        
        with open(config_path, 'w') as f:
            json.dump(test_data, f)
        
        result = collector.load_symbol_map(str(config_path))
        assert result == test_data
    
    def test_load_missing_config_returns_defaults(self, tmp_path):
        """Test that missing config returns default values."""
        config_path = tmp_path / "nonexistent.json"
        result = collector.load_symbol_map(str(config_path))
        
        assert "BTCUSD" in result
        assert "EURUSD" in result
        assert "XAUUSD" in result


class TestInitializeMT5:
    """Tests for MT5 initialization."""
    
    def test_initialize_success(self):
        """Test successful MT5 initialization."""
        mock_mt5.initialize.return_value = True
        result = collector.initialize_mt5()
        assert result is True
    
    def test_initialize_failure(self):
        """Test failed MT5 initialization."""
        mock_mt5.initialize.return_value = False
        result = collector.initialize_mt5()
        assert result is False


class TestFetchCandles:
    """Tests for fetch_candles function."""
    
    def test_fetch_success(self):
        """Test successful candle fetch."""
        # Mock rates data
        mock_rates = np.array([
            (1609459200, 1.2000, 1.2100, 1.1900, 1.2050, 1000, 0, 0),
            (1609459260, 1.2050, 1.2150, 1.1950, 1.2100, 1200, 0, 0),
        ], dtype=[
            ('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'),
            ('close', 'f8'), ('tick_volume', 'i8'), ('spread', 'i8'), ('real_volume', 'i8')
        ])
        
        mock_mt5.copy_rates_from_pos.return_value = mock_rates
        
        df = collector.fetch_candles("EURUSD", 2)
        
        assert df is not None
        assert len(df) == 2
        assert "time" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
    
    def test_fetch_no_data(self):
        """Test fetch with no data returned."""
        mock_mt5.copy_rates_from_pos.return_value = None
        
        df = collector.fetch_candles("INVALID", 100)
        
        assert df is None
    
    def test_fetch_empty_data(self):
        """Test fetch with empty data."""
        mock_mt5.copy_rates_from_pos.return_value = np.array([])
        
        df = collector.fetch_candles("EURUSD", 100)
        
        assert df is None


class TestUpdateParquet:
    """Tests for update_parquet function."""
    
    def test_new_file_creation(self, tmp_path):
        """Test creating a new parquet file."""
        # Setup
        collector.DATA_DIR = str(tmp_path)
        
        mock_rates = np.array([
            (1609459200, 1.2000, 1.2100, 1.1900, 1.2050, 1000, 0, 0),
            (1609459260, 1.2050, 1.2150, 1.1950, 1.2100, 1200, 0, 0),
        ], dtype=[
            ('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'),
            ('close', 'f8'), ('tick_volume', 'i8'), ('spread', 'i8'), ('real_volume', 'i8')
        ])
        
        mock_mt5.copy_rates_from_pos.return_value = mock_rates
        mock_mt5.symbol_select.return_value = True
        
        # Execute
        result = collector.update_parquet("TEST", "TEST", max_candles=1000)
        
        # Verify
        assert result is True
        expected_path = tmp_path / "TEST_M1.parquet"
        assert expected_path.exists()
        
        # Verify content
        df = pd.read_parquet(expected_path)
        assert len(df) == 2
        assert "time" in df.columns
    
    def test_symbol_not_available(self, tmp_path):
        """Test handling when symbol is not available."""
        collector.DATA_DIR = str(tmp_path)
        mock_mt5.symbol_select.return_value = False
        
        result = collector.update_parquet("INVALID", "INVALID")
        
        assert result is False


class TestCollectAssets:
    """Tests for collect_assets function."""
    
    def test_collect_multiple_assets(self, tmp_path):
        """Test collecting multiple assets."""
        collector.DATA_DIR = str(tmp_path)
        
        mock_rates = np.array([
            (1609459200, 1.2000, 1.2100, 1.1900, 1.2050, 1000, 0, 0),
        ], dtype=[
            ('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'),
            ('close', 'f8'), ('tick_volume', 'i8'), ('spread', 'i8'), ('real_volume', 'i8')
        ])
        
        mock_mt5.copy_rates_from_pos.return_value = mock_rates
        mock_mt5.symbol_select.return_value = True
        
        # Create temp config
        config_path = tmp_path / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump({"TEST1": "TEST1", "TEST2": "TEST2"}, f)
        
        results = collector.collect_assets(
            ["TEST1", "TEST2"],
            config_path=str(config_path)
        )
        
        assert results["TEST1"] is True
        assert results["TEST2"] is True
    
    def test_collect_unknown_asset(self, tmp_path):
        """Test collecting an unknown asset."""
        collector.DATA_DIR = str(tmp_path)
        
        config_path = tmp_path / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump({"KNOWN": "KNOWN"}, f)
        
        results = collector.collect_assets(
            ["UNKNOWN"],
            config_path=str(config_path)
        )
        
        assert results["UNKNOWN"] is False


class TestDeduplication:
    """Tests for deduplication logic."""
    
    def test_deduplication_on_time(self, tmp_path):
        """Test that duplicate timestamps are removed."""
        collector.DATA_DIR = str(tmp_path)
        
        # Create existing data
        existing_df = pd.DataFrame({
            "time": pd.to_datetime(["2021-01-01 00:00:00", "2021-01-01 00:01:00"], utc=True),
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "tick_volume": [100, 200],
        })
        
        existing_path = tmp_path / "DEDUP_M1.parquet"
        existing_df.to_parquet(existing_path, index=False)
        
        # Mock new data with overlapping timestamp
        mock_rates = np.array([
            (1609459260, 1.2050, 1.2150, 1.1950, 1.2100, 1200, 0, 0),  # Same as existing
            (1609459320, 1.2100, 1.2200, 1.2000, 1.2150, 1300, 0, 0),  # New
        ], dtype=[
            ('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'),
            ('close', 'f8'), ('tick_volume', 'i8'), ('spread', 'i8'), ('real_volume', 'i8')
        ])
        
        mock_mt5.copy_rates_from_pos.return_value = mock_rates
        mock_mt5.symbol_select.return_value = True
        
        collector.update_parquet("DEDUP", "DEDUP")
        
        # Verify deduplication
        df = pd.read_parquet(existing_path)
        assert len(df) == 3  # 2 existing + 1 new (1 duplicate removed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
