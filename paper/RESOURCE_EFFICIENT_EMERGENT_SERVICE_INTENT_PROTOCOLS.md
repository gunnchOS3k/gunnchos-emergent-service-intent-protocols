# Resource-Efficient Emergent Service-Intent Protocols for Distributed AI-RAN Agents

**Evidence class for all reported smoke numbers in this repository: `SYNTHETIC_SIM` (CPU).**  
GPU / over-the-air claims are out of scope until labeled hardware runs exist. DOI: **DOI_PENDING**.

## Abstract

We study multi-agent reinforcement learning (MARL) of communication protocols under validated **service-intent** constraints in terrestrial and TN–NTN AI-RAN settings. Agents (UE, BS, orchestrator, optional NTN relay) act in a Doc-POMDP with partial observability and costly, unreliable messaging. We provide open implementations of IPPO, MAPPO, VDN/QMIX, and a DIAL/TarMAC-style communication-aware baseline, plus neuro-symbolic intent parsing that never grants an LLM direct radio control.

## Mathematical model

See `docs/MATH_FORMULATION.md`. In brief, agents $i\in\mathcal{I}$ maximize expected scalarized return
$$
J(\pi) = \mathbb{E}\Big[\sum_{t=0}^{T-1} R_t(s_t,a_t,m_t)\Big]
$$
with $R_t$ from weighted utilities over task success, latency/AoI, energy, message bits, fairness, spectral efficiency, and constraint violations. Discrete messages use vocabulary size $V$, length $L$, bit cost $c_b$, silence, erasures, corruption, and delay. Training may use Gumbel-Softmax; execution uses hard symbols.

## Related work matrix

| Theme | Representative lines | Gap addressed here |
|------|----------------------|--------------------|
| Emergent communication | DIAL, RIAL, TarMAC, IC3Net | Bit-cost + RAN constraints + intent safety |
| MARL cooperative control | IPPO, MAPPO, QMIX/VDN | Doc-POMDP AI-RAN scenarios |
| Intent-based networking | IETF/3GPP intent, SDN intents | Neuro-symbolic parse → action masks / Lagrangian |
| NTN resilience | TN–NTN handover sims | Coupled with learned messaging |
| Interpretability | topographic similarity, MI probes | Required before “language” claims |

## Hypotheses

1. **H1 (resource efficiency):** Under fixed task utility, discrete learned protocols with bit cost achieve lower message bits than continuous free-channel baselines.
2. **H2 (failover):** Targeted discrete messaging improves TN–NTN failover task success vs no-comm under partial observability.
3. **H3 (fairness):** Preference / Lagrangian objectives raise fairness in education scenarios without catastrophic latency regressions.
4. **H4 (intent safety):** Rule-parsed intents with action masks reduce violation rate vs unconstrained policies.

All hypotheses are tested only on `SYNTHETIC_SIM` unless a labeled hardware profile is attached.

## Baselines and ablations

- Communication: `no_comm`, `fixed_protocol`, `continuous_learned`, `discrete_learned` (± targeted)
- Algorithms: IPPO, MAPPO, VDN, QMIX, DIAL/TarMAC
- Abstraction: raw / engineered / IB / VQ / contrastive
- Objectives: weighted / preference / Lagrangian
- Channel impairments: erasure, corruption, delay sweeps

## Statistical protocol

CPU smoke tables use **≥5 seeds** for the primary IPPO config. Report mean return and eval return; do not invent GPU numbers. Medium CPU runs are labeled separately.

## Limitations

- Dynamics are synthetic abstractions, not calibrated cell traces.
- Apple M2 runs are CPU-only; CUDA paths are detected and blocked honestly when absent.
- Interpretability metrics do **not** alone justify linguistic claims.
- Sibling repo adapters are soft and may be absent.

## Reproducibility

```bash
make bootstrap && make test && make smoke && make gate4-cpu
make gate4-gpu  # expects BLOCKED_HARDWARE without CUDA
```

Configs live under `configs/`; manifests under `results/smoke/` include seed, commit, device label, and evidence class.

## Status flags

- RELEASE_CANDIDATE_READY (CPU)
- DOI_PENDING
- INDEPENDENT_REPRODUCTION_PENDING
- PEER_REVIEW_PENDING
