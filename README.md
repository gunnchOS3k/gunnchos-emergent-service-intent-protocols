# Emergent Service-Intent Protocols

Research codebase for **resource-efficient communication protocols** among distributed AI-RAN agents under validated **service-intent** constraints.

Aligned to the **public** University of Oulu GENOME research theme (emergent communication in beyond 5G/6G). **Not an appointment, studentship, or funded post.** Optional distributed-intelligence extension — **not** a fourth dissertation paper.

Implements a Decentralized Partially Observable MDP with communication (**Doc-POMDP**) for UE/device, BS/RAN, edge/cloud orchestrator, and optional NTN relay agents.

Supervisor path: [`docs/START_HERE_SUPERVISOR.md`](docs/START_HERE_SUPERVISOR.md) ·
[`docs/CLAIM_BOUNDARIES.md`](docs/CLAIM_BOUNDARIES.md) ·
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) ·
[`docs/uml/README.md`](docs/uml/README.md).

Private clone: [`docs/packets/REPOSITORY_VISIBILITY_PACKET.md`](docs/packets/REPOSITORY_VISIBILITY_PACKET.md).

## Truth rules

- Never fabricate physical RF or GPU acceleration results.
- Default host evidence is **CPU-only** (`SYNTHETIC_SIM`).
- `make gate4-gpu` exits honestly with `BLOCKED_GPU` when CUDA is absent.
- Do not claim “emergent language” without interpretability analysis (entropy, MI, topographic similarity, interventions).

## Status

| Flag | Status |
|------|--------|
| RELEASE_CANDIDATE_READY | candidate (CPU smoke + tests) |
| DOI_PENDING | pending |
| INDEPENDENT_REPRODUCTION_PENDING | pending |
| PEER_REVIEW_PENDING | pending |

## Quick start

```bash
make bootstrap
make test
make smoke
make gate4-cpu
make gate4-gpu   # BLOCKED_GPU on Apple Silicon without CUDA
```

## Layout

See `src/emergent_intent/` for `env`, `comm`, `algorithms`, `abstraction`, `objectives`, `intent`, `interpretability`, `adapters`.

## Citation

See `CITATION.cff`. Manuscript: `paper/RESOURCE_EFFICIENT_EMERGENT_SERVICE_INTENT_PROTOCOLS.md`.

## License

Apache-2.0
