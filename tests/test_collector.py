import pytest
import pandas as pd
from unittest.mock import patch
import json
from pathlib import Path

# Import the actual functions (no DataCollector class anymore)
from data_layer.collector import (
    load_symbol_map,
    initialize_mt5,
    fetch_candles,
    update_parquet,
    collect_assets,
    deduplicate,
)


@pytest.fixture
def mock_mt5():
    """Correct patch target — this is the alias used inside collector.py"""
    with patch("data_layer.collector.mt5") as mock:
        mock.initialize.return_value = True
        mock.symbol_select.return_value = True
        mock.copy_rates_from_pos.return_value = [
            {
                "time": 1743100800,
                "open": 2650.0,
                "high": 2655.0,
                "low": 2648.0,
                "close": 2652.0,
                "tick_volume": 1200,
            }
        ] * 50
        yield mock


def test_load_existing_config(tmp_path):
    config = {"XAUUSD": "GOLD", "BTCUSD": "BITCOIN"}
    path = tmp_path / "symbol_map.json"
    path.write_text(json.dumps(config))
    result = load_symbol_map(str(path))
    assert result == {"XAUUSD": "GOLD", "BTCUSD": "BITCOIN"}


def test_load_missing_config_returns_defaults():
    result = load_symbol_map(None)  # or missing path
    assert "XAUUSD" in result
    assert isinstance(result, dict)


def test_initialize_success(mock_mt5):
    result = initialize_mt5()
    assert result is True
    mock_mt5.initialize.assert_called_once()


def test_initialize_failure(mock_mt5):
    mock_mt5.initialize.return_value = False
    with pytest.raises(RuntimeError, match="MT5 initialization failed"):
        initialize_mt5()


def test_fetch_success(mock_mt5):
    df = fetch_candles("XAUUSD", 100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_fetch_no_data(mock_mt5):
    mock_mt5.copy_rates_from_pos.return_value = None
    with pytest.raises(RuntimeError, match="No data returned"):
        fetch_candles("INVALID", 100)


def test_fetch_empty_data(mock_mt5):
    mock_mt5.copy_rates_from_pos.return_value = []
    with pytest.raises(RuntimeError, match="No data returned"):
        fetch_candles("EURUSD", 100)


def test_update_parquet_new_file_creation(mock_mt5, tmp_path):
    with patch("data_layer.collector.DATA_DIR", str(tmp_path)):
        result = update_parquet("XAUUSD", "GOLD")
        assert result is True
        assert (tmp_path / "GOLD_M1.parquet").exists()


def test_update_parquet_symbol_not_available(mock_mt5):
    mock_mt5.symbol_select.return_value = False
    with pytest.raises(RuntimeError, match="Failed to select symbol"):
        update_parquet("INVALID", "INVALID")


def test_collect_assets(mock_mt5, tmp_path):
    with patch("data_layer.collector.DATA_DIR", str(tmp_path)):
        results = collect_assets(["XAUUSD"])
        assert isinstance(results, dict)
        assert "XAUUSD" in results


def test_deduplication_on_time():
    df = pd.DataFrame({"time": [1, 1, 2], "close": [100, 100, 101]})
    result = deduplicate(df)
    assert len(result) == 2
