# GENOME smoke notebook (placeholder for interactive exploration)

Use `make smoke` for canonical seeded runs. This notebook is optional.

```python
from emergent_intent.env import make_env, EnvConfig
env = make_env(EnvConfig(horizon=8))
obs, info = env.reset(seed=0)
print(list(obs))
```
