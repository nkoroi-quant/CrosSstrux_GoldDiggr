# run_training_pipeline.py - Legacy wrapper kept but marked deprecated

import warnings
warnings.warn("run_training_pipeline.py is deprecated - use training.train directly or make train", DeprecationWarning)

from training.train import train_asset, main as _main  # keep exact API

# mirror all functions
if __name__ == "__main__":
    _main()