# Reproducibility — emergent service-intent protocols

CPU-first Doc-POMDP for service-intent constrained multi-agent communication. Default evidence: `SYNTHETIC_SIM`.

## Fresh machine (CPU)

```bash
git clone https://github.com/gunnchOS3k/gunnchos-emergent-service-intent-protocols.git
cd gunnchos-emergent-service-intent-protocols
# Private clone requires collaborator access (docs/packets/REPOSITORY_VISIBILITY_PACKET.md).
python3 -m pip install -e ".[dev]"
make test
make gate4-cpu
make blocked-gpu          # BLOCKED_GPU JSON without CUDA
make supervisor-cpu-gate
```

Python 3.11+, PyTorch CPU wheels. GPU: `make gate4-gpu` writes `BLOCKED_GPU` when CUDA is absent.

## Expected CPU outputs

- pytest PASS (unit + scientific claim/efficiency/repeated-run tests).
- `GATE4_OULU_CPU_OK (smoke≠scientific final)` from `make gate4-cpu`.
- `results/blocked_gpu/BLOCKED_GPU.json` with `status: BLOCKED_GPU` on Apple Silicon.
- `results/supervisor/SUPERVISOR_CPU_GATE.json` with `emergent_language_claimed: false`.

## What not to conclude

- Message traffic is not an emergent language.
- Smoke returns are not the final experiment matrix.
- GENOME in the title is a public research theme, not a job held.

Record `git rev-parse HEAD` in any supervisor packet.
