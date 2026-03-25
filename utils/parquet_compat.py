# utils/parquet_compat.py

import pandas as pd

_original_read_parquet = pd.read_parquet


def install():
    def patched(*args, **kwargs):
        kwargs["engine"] = "pyarrow"
        return _original_read_parquet(*args, **kwargs)

    pd.read_parquet = patched


def read_parquet(path):
    return _original_read_parquet(path, engine="pyarrow")


def write_parquet(df, path):
    df.to_parquet(path, index=False, engine="pyarrow")
