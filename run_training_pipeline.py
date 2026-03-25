import warnings

warnings.warn(
    "run_training_pipeline.py is deprecated. Use `make train` or `python -m training.train` instead",
    DeprecationWarning,
)

from training.train import main

if __name__ == "__main__":
    main()
