# Resource-Efficient Emergent Service-Intent Protocols for Distributed AI-RAN Agents

**Evidence class for all numerical results in this manuscript: `SYNTHETIC_SIM` (CPU pilot/smoke).**  
GPU / over-the-air / independent reproduction: **not earned**. DOI: **DOI_PENDING**.

## 1. Abstract

We study multi-agent reinforcement learning (MARL) of resource-efficient communication protocols under validated service-intent constraints in terrestrial and TN–NTN AI-RAN settings. Agents (UE, BS, edge orchestrator, optional NTN relay) act in a Documented POMDP (Doc-POMDP) with partial observability and costly, unreliable messaging. This corrective release repairs critical scientific gaps: receiver observations now include bounded message inboxes; radio actions causally affect service outcomes; communication-necessary scenarios expose information asymmetry; QMIX/VDN use replay, target networks, and Double Q; and a previously mislabeled “DIAL/TarMAC” trainer is renamed to `ppo_discrete_message_entropy_baseline`, while separate faithful DIAL and TarMAC trainers provide gradient-through-channel and attention-routing evidence under pilot budgets. Reported numbers are pilot-scale (`SYNTHETIC_SIM`); they are not final scientific claims.

## 2. Introduction

Distributed AI-RAN control must coordinate under bandwidth, energy, and intent constraints. Learned messaging can help when local observations are incomplete, but only if messages enter policy inputs and actions actually change service probability. Prior scaffolding in this repository stored messages without inbox features and computed delivery largely from queue thresholds—invalidating communication and control claims. This paper documents the corrected environment, algorithms, pilot matrix, and remaining gaps.

## 3. Related work

| Theme | Representative lines | Gap addressed here |
|------|----------------------|--------------------|
| Emergent communication | DIAL, RIAL, TarMAC, IC3Net | Bit-cost + RAN constraints + honest naming |
| MARL cooperative control | IPPO, MAPPO, QMIX/VDN | Causal Doc-POMDP AI-RAN scenarios |
| Intent-based networking | IETF/3GPP intent forms | Neuro-symbolic parse → masks / Lagrangian |
| NTN resilience | TN–NTN handover sims | Coupled with messaging under asymmetry |
| Interpretability | MI probes, topographic similarity | Estimates only; no language claim |

## 4. System model

Agents $\mathcal{I}$ include UEs, one BS, one edge orchestrator, and optional NTN relay. Continuous wireless state tracks queues, AoI, SNR proxies, blockage, congestion, TN/NTN availability, fairness debt, interference, and intent flags. Actions factor into power, PRB allocation, MCS, access, handover, routing, admission, priority, offload, discrete message token, and target index. Service probability depends on power, PRB, MCS, link quality (TN/NTN), congestion, interference, priority, coordination from messages, and admission—not on a queue-threshold alone.

## 5. Doc-POMDP formulation

Each agent $i$ receives observation $o^i_t = [\,o^{i,\mathrm{local}}_t;\; m^{i,\mathrm{in}}_t\,]$ where inbox slots carry sender embedding, symbols, age, staleness, confidence, erasure, silence, and validity. Transitions and rewards form a cooperative Doc-POMDP with team scalarization over task success, AoI, energy, message bits, fairness, spectral efficiency, and violations. A centralized oracle may read global $s_t$ only as an upper-bound baseline.

## 6. Communication architecture

Modes: `no_comm`, `fixed_protocol`, `continuous_learned`, `discrete_learned`. Discrete channels support vocabulary $V$, length $L$, silence, erasure, corruption, delay queues, targeted delivery, loopback (off by default), and bounded inbox capacity. Unit tests 1–10 (§6.1) verify inbox causality. Communication-necessary Scenario A hides UE blockage from the BS while exposing congestion; Scenario B requires TN/NTN continuity coordination across asymmetric local views.

## 7. State abstraction

Operational paths feeding the policy: raw observations; engineered aggregation; information-bottleneck encoder; vector quantization; contrastive projector. Pilot reports under `results/ablations/abstraction_pilot.json` record latent size, objective, regularization, downstream utility, message efficiency, held-out return, stability, nuisance sensitivity, and compute cost. These are pilot-scale and not final abstraction rankings.

## 8. Multi-objective method

We compare fixed weighted scalarization, preference-conditioned weights, and Lagrangian penalization of violations/bits/energy. Pareto fronts are written to `results/pilot/multiobj/pareto_front.json` and plotted in `paper/figures/pareto_front.png` when pilot data exist. Objectives reported: task success, AoI/latency, energy, message bits, fairness, spectral efficiency, violations.

## 9. Algorithms

| Name | Role | Honest status |
|------|------|----------------|
| random / no-comm / fixed protocol | baselines | implemented |
| IPPO / MAPPO | on-policy MARL | validated unit tests beyond non-crash |
| VDN / QMIX | value factorization | replay, targets, Double Q, terminal mask, checkpoint |
| `ppo_discrete_message_entropy_baseline` | discrete-message PPO + entropy | truthful rename of former DIAL/TarMAC label |
| DIAL | Gumbel train / hard eval; receiver loss → sender msg-head | gradient test required & present |
| TarMAC | learned attention routing into receiver policy | attention diagnostics recorded |
| centralized oracle | upper bound | heuristic coordinated actions |

## 10. Experimental setup

CPU-only synthetic Doc-POMDP. Smoke: short configs under `configs/smoke/` (≤512 steps). Pilot matrix: algorithms × scenarios, default ≥5 seeds when compute allows (`scripts/run_pilot_experiments.py`). Result directories: `results/{smoke,pilot,final,ablations,generalization,robustness,interpretability}/`. **Final experiments are explicitly not claimed** for 64–512 step runs; `results/final/STATUS.json` records `NOT_RUN`.

## 11. Main results

Pilot returns (when generated) live in `results/pilot/pilot_summary.json` and `paper/figures/pilot_returns.png`. Evidence label: `SYNTHETIC_SIM` / `PILOT`. Causal environment tests show power and PRB increase expected service; wrong TN selection under outage hurts; messaging improves the hidden-blockage scenario; oracle dominates weak randomish controls. **No algorithm is declared scientifically superior for admission or deployment.**

## 12. Ablations

Abstraction pilots and communication-mode comparisons are filed under `results/ablations/`. Removing messages in Scenario A reduces mean served traffic in causal tests. Full factorial ablations (vocab, delay, inbox capacity) remain incomplete.

## 13. Generalization

Held-out UE counts (12–16) and digitial-twin layout sweeps are **not complete**. `results/generalization/STATUS.json` is `PILOT_PARTIAL`. This manuscript does not claim cross-topology generalization.

## 14. Robustness

Unit tests cover erasure, corruption, delay, silence, and reset hygiene. A full robustness matrix (agent dropout, reward misspecification, domain shift) is only partially automated; status `PILOT_PARTIAL`.

## 15. Interpretability

`results/interpretability/interpretability_probe.json` reports message entropy, symbol utilization, histogram MI estimates, topographic similarity, and symbol–condition matrices for short rollouts. These are **estimates only**. We do **not** claim an emergent language.

## 16. Negative results

- Former DIAL/TarMAC naming was false; gradients did not flow through a communication channel. Corrected by rename + new trainers.
- Smoke non-crash success does not imply scientific PASS (status-integrity tests enforce this).
- Pilot seed matrices may still skip some algorithm×scenario cells under `--quick` CI settings.

## 17. Limitations

Dynamics are synthetic abstractions, not calibrated cell traces. Apple Silicon / CPU hosts cannot produce CUDA evidence. Intent adapters never emit radio actions directly but coverage of natural-language intents is narrow. Independent reproduction and DOI remain pending.

## 18. Ethics and privacy

No human-subject data. Synthetic queues/SNR only. Intent text is rule-parsed with rejection of ambiguous commands; LLMs (if used) cannot directly set radio actions. Dual-use radio control research should not be over-claimed as deployable autonomy.

## 19. Reproducibility

```bash
make bootstrap
make test
make causal-tests
make algorithm-validation
make smoke
make final-experiments   # writes pilot/; does not claim final
make statistics
make figures
make paper
make artifact
make reproduce-clean
```

Manifests include commit SHA, seed, device label, and `evidence_class`. Clean-checkout reproduction may earn internal reproduction only when harnesses pass; **independent reproduction is pending**.

## 20. Conclusion

Corrective-depth work restored causal and communicational validity for this Doc-POMDP core, separated truthful baselines from faithful DIAL/TarMAC, upgraded value-decomposition training, and recorded pilot statistics without fabricating final or hardware evidence. Remaining blockers: large-seed final matrix, full generalization/robustness sweeps, GPU measurements, physical pilots, DOI, and independent review.
