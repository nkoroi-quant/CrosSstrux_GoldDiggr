# Models directory

CrossStrux v2 does **not** ship trained models in this package.

After training, each asset gets its own folder here:

```text
models/XAUUSD/
├── metadata.json
├── low_model.pkl
├── mid_model.pkl
├── high_model.pkl
├── transition_model.pkl
├── drift_baseline_low.json
├── drift_baseline_mid.json
└── drift_baseline_high.json
```

Run the training pipeline after ingesting raw XAUUSD data.
