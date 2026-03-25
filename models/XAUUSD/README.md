# XAUUSD model folder

This folder is intentionally empty in the source-only package.

Train the model stack with:

```bash
python run_training_pipeline.py --assets XAUUSD --force-retrain
```

The training script will create `metadata.json`, regime models, transition model,
and drift baselines in this folder.
