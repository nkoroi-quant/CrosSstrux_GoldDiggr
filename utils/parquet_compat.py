# utils/parquet_compat.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_original_df_to_parquet = pd.DataFrame.to_parquet
_original_read_parquet = pd.read_parquet
_installed = False


def _native_parquet_available() -> bool:
    try:
        import pyarrow  # type: ignore  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import fastparquet  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


_HAS_NATIVE = _native_parquet_available()


def _write_fallback_pickle(df: pd.DataFrame, path: str | Path, *args: Any, **kwargs: Any) -> None:
    df.to_pickle(path)


def _read_fallback_pickle(path: str | Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
    return pd.read_pickle(path)


def install() -> None:
    """Install a parquet compatibility shim.

    If a native parquet engine is available, we keep using pandas' real parquet
    methods. Otherwise we transparently fall back to pickle storage so the rest
    of the project can keep using the .parquet extension in tests/dev.
    """
    global _installed
    if _installed:
        return

    if _HAS_NATIVE:
        def patched_to_parquet(self: pd.DataFrame, path, *args, **kwargs):
            return _original_df_to_parquet(self, path, *args, **kwargs)

        def patched_read_parquet(path, *args, **kwargs):
            return _original_read_parquet(path, *args, **kwargs)
    else:
        def patched_to_parquet(self: pd.DataFrame, path, *args, **kwargs):
            return _write_fallback_pickle(self, path, *args, **kwargs)

        def patched_read_parquet(path, *args, **kwargs):
            return _read_fallback_pickle(path, *args, **kwargs)

    pd.DataFrame.to_parquet = patched_to_parquet  # type: ignore[assignment]
    pd.read_parquet = patched_read_parquet  # type: ignore[assignment]
    _installed = True


# Install immediately so direct DataFrame.to_parquet() calls work in tests/dev.
install()


def read_parquet(path):
    return pd.read_parquet(path)


def write_parquet(df, path):
    return df.to_parquet(path, index=False)
