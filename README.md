# Venik 🧹
Sweeps-like wrapper around MLflow and Optuna

# Sweeps
Run a sweep:
```bash
python -m venik.sweep sweep.yaml
```

Run an agent:
```bash
python -m venik.agent <sweep-id> [-a arg1 arg2 ...]
```

# Logger
The training script must use the following logger, compatible with PyTorch Lightning:
```python
from venik import MLFlowLogger
```

# Extra MLFlow tools
Download parameters:
```bash
python -m venik.api parameters <run-id> > parameters.yaml
```

Download metrics:
```bash
python -m venik.api metrics <run-id> > metrics.yaml
```
