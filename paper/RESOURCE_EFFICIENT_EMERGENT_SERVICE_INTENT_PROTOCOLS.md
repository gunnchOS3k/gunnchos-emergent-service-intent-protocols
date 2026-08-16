# Resource-Efficient Emergent Service-Intent Protocols for Distributed AI-RAN Agents

**Evidence class for all numerical results in this manuscript: `SYNTHETIC_SIM` (CPU).**  
GPU / over-the-air / independent reproduction: **not earned**. DOI: **DOI_PENDING**.

## 1. Abstract

We study multi-agent reinforcement learning (MARL) of resource-efficient communication protocols under validated service-intent constraints in terrestrial and TN–NTN AI-RAN settings. Agents (UE, BS, edge orchestrator, optional NTN relay) act in a Documented POMDP (Doc-POMDP) with partial observability and costly, unreliable messaging. This publication-grade Track B release removes direct message-presence reward bonuses so messages help only via observations→actions; adds semantic intervention tests (correct/random/constant/permute/silence/delay/stale/corrupt/adversarial) with fixed-protocol mappings for blockage, congestion, TN/NTN, priority, and handover; and validates faithful DIAL (differentiable soft-channel train → hard eval with **task-loss** gradients into the sender message-head) and faithful TarMAC (real peer key/value/query/attention stored in rollouts; joint logπ over control+message+target; no self-tiling as peers). Reported numbers are `SYNTHETIC_SIM`; where wall-clock budgets truncate the matrix we label `BLOCKED_COMPUTE_CAPACITY` rather than fabricating finals.

## 2. Introduction

Distributed AI-RAN control must coordinate under bandwidth, energy, and intent constraints. Learned messaging can help when local observations are incomplete, but only if messages enter policy inputs and actions actually change service probability—without shortcut presence bonuses in the reward. This paper documents the corrected environment, faithful DIAL/TarMAC trainers, semantic intervention evidence, final/ablation/generalization/robustness directories, and remaining blockers.

## 3. Related work

| Theme | Representative lines | Gap addressed here |
|------|----------------------|--------------------|
| Emergent communication | DIAL, RIAL, TarMAC, IC3Net | Bit-cost + RAN constraints + honest naming |
| MARL cooperative control | IPPO, MAPPO, QMIX/VDN | Causal Doc-POMDP AI-RAN scenarios |
| Intent-based networking | IETF/3GPP intent forms | Neuro-symbolic parse → masks / Lagrangian |
| NTN resilience | TN–NTN handover sims | Coupled with messaging under asymmetry |
| Interpretability | MI probes, topographic similarity | Estimates only; no language claim |

## 4. System model

Agents $\mathcal{I}$ include UEs, one BS, one edge orchestrator, and optional NTN relay. Continuous wireless state tracks queues, AoI, SNR proxies, blockage, congestion, TN/NTN availability, fairness debt, interference, and intent flags. Actions factor into power, PRB allocation, MCS, access, handover, routing, admission, priority, offload, discrete message token, and target index. Service probability depends on power, PRB, MCS, link quality (TN/NTN), congestion, interference, priority, admission, and control actions that may be conditioned on inbox observations—not on a queue-threshold alone and **not** on a direct message-presence reward bonus. Rewards scalarize task success, latency/AoI, energy, message-bit cost, fairness, spectral efficiency, and violations only.

## 5. Doc-POMDP formulation

Each agent $i$ receives observation $o^i_t = [\,o^{i,\mathrm{local}}_t;\; m^{i,\mathrm{in}}_t\,]$ where inbox slots carry sender embedding, symbols, age, staleness, confidence, erasure, silence, and validity. Transitions and rewards form a cooperative Doc-POMDP with team scalarization over the objectives above. A centralized oracle may read global $s_t$ only as an upper-bound baseline.

## 6. Communication architecture

Modes: `no_comm`, `fixed_protocol`, `continuous_learned`, `discrete_learned`. Discrete channels support vocabulary $V$, length $L$, silence, erasure, corruption, delay queues, targeted delivery, loopback (off by default), and bounded inbox capacity. Fixed-protocol encoders emit canonical symbols for blockage, congestion, TN/NTN, priority, and handover. A `SemanticProtocolController` maps inbox symbols to radio actions (obs→act). Semantic intervention tests under `tests/scientific/test_message_semantic_causality.py` show destructive interventions (silence/adversarial/random/constant) reduce return relative to the correct protocol without enabling presence bonuses.

## 7. State abstraction

Operational paths feeding the policy: raw observations; engineered aggregation; information-bottleneck encoder; vector quantization; contrastive projector. Ablation artifacts under `results/ablations/` record pilot abstraction probes and communication-mode comparisons when generated.

## 8. Multi-objective method

We compare fixed weighted scalarization, preference-conditioned weights, and Lagrangian penalization of violations/bits/energy. Objectives reported: task success, AoI/latency, energy, message bits, fairness, spectral efficiency, violations.

## 9. Algorithms

| Name | Role | Honest status |
|------|------|----------------|
| random / no-comm / fixed protocol / oracle | baselines | implemented; complete list in `scripts/run_final_experiments.py` |
| IPPO / MAPPO | on-policy MARL | unit-validated beyond non-crash |
| VDN / QMIX | value factorization | replay, targets, Double Q, terminal mask, checkpoint |
| `ppo_discrete_message_entropy_baseline` | discrete-message PPO + entropy | truthful rename of former DIAL/TarMAC label |
| DIAL | soft Gumbel train / hard eval; **task loss** → sender msg-head | end-to-end tests in `tests/scientific/test_dial_end_to_end.py` |
| TarMAC | real peer KVQ/attention in rollout; joint logπ | end-to-end tests in `tests/scientific/test_tarmac_end_to_end.py` |

## 10. Experimental setup

CPU-only synthetic Doc-POMDP. Smoke: ≤512 steps under `configs/smoke/`. Final driver: `scripts/run_final_experiments.py` with flagship budgets **beyond 512 steps** (default 1024) and **5 seeds** where wall-clock allows. Result directories: `results/{smoke,pilot,final,ablations,generalization,robustness,interpretability,interventions}/`. If the wall-clock budget truncates the matrix, `STATUS.json` records `BLOCKED_COMPUTE_CAPACITY` with the completed subset—never a fabricated full PASS.

## 11. Main results

**Final matrix (`results/final/STATUS.json`: `RAN`, 1024 steps, 5 seeds × 3 scenarios × 9 methods, `SYNTHETIC_SIM`, CPU-only Apple Silicon).** Aggregated mean episodic returns (n=15 cells per method; ± sample std):

| Method | Mean return | Std | 5-seed eligible |
|--------|------------:|----:|:---------------:|
| fixed_protocol (semantic obs→act) | −0.47 | 16.74 | yes |
| TARMAC | −10.48 | 12.55 | yes |
| ppo_discrete_message_entropy_baseline | −11.22 | 12.51 | yes |
| IPPO | −12.21 | 12.44 | yes |
| random | −12.96 | 10.62 | yes |
| DIAL | −13.30 | 13.11 | yes |
| MAPPO | −13.40 | 11.95 | yes |
| no_comm | −17.01 | 12.13 | yes |
| QMIX | −23.09 | 15.64 | yes |

**Semantic interventions** (`results/interventions/`, hidden-blockage, 5 seeds): correct protocol mean return **+7.18**; silence **−13.98**; adversarial **−14.01**; random **−7.14**; corrupt **−2.45**; delay **+3.05**. Presence-bonus coordination in infos remains **0.0**. **No algorithm is declared scientifically superior for admission or deployment**; returns are short-horizon synthetic and high-variance.

## 12. Ablations

Ablation suite on hidden-blockage (`results/ablations/STATUS.json`: `RAN`, 5 seeds): fixed_protocol −16.82; ppo-message-entropy −22.71; DIAL −24.73; no_comm −25.42. Removing usable inbox content (silence intervention) reduces return under the semantic controller. Full factorial vocab/delay/inbox sweeps remain incomplete.

## 13. Generalization

Held-out UE-count sweeps (1–3) ran under `results/generalization/STATUS.json`: `RAN` (DIAL mean −24.00 over n=15). Larger counts (12–16) and digital-twin layout sweeps are **not complete**. This manuscript does not claim cross-topology generalization.

## 14. Robustness

Erasure-rate sweeps under fixed-protocol semantic control: `results/robustness/STATUS.json` `RAN` (fixed_protocol mean +5.42 over n=15 across erasure ∈ {0,0.2,0.4}). Unit tests cover erasure, corruption, delay, silence, and reset hygiene. Full domain-shift / agent-dropout matrices remain partial.

## 15. Interpretability

`results/interpretability/interpretability_probe.json` reports message entropy, symbol utilization, histogram MI estimates, topographic similarity, and symbol–condition matrices for short rollouts. These are **estimates only**. We do **not** claim an emergent language.

## 16. Negative results

- Message-presence bonuses were a scientific shortcut; they are disabled by default (`message_presence_bonus_enabled=False`).
- Former DIAL/TarMAC naming was false; corrected by rename + faithful trainers with end-to-end tests.
- Receiver-value regression is **not** the primary DIAL objective; task loss through the channel is.
- TarMAC must not tile self as peers; update refuses missing peer stores.
- Smoke non-crash success does not imply scientific PASS (status-integrity tests enforce this).
- Compute budgets may leave `BLOCKED_COMPUTE_CAPACITY` rather than a complete 5-seed×all-methods final matrix.

## 17. Limitations

Dynamics are synthetic abstractions, not calibrated cell traces. Apple Silicon / CPU hosts cannot produce CUDA evidence. Intent adapters never emit radio actions directly but coverage of natural-language intents is narrow. Independent reproduction and DOI remain pending.

## 18. Ethics and privacy

No human-subject data. Synthetic queues/SNR only. Intent text is rule-parsed with rejection of ambiguous commands; LLMs (if used) cannot directly set radio actions. Dual-use radio control research should not be over-claimed as deployable autonomy.

## 19. Reproducibility

```bash
make bootstrap
make test
make causal-tests
make semantic-intervention-tests
make dial-validation
make tarmac-validation
make final-experiments
make generalization
make robustness
make ablations
make interpretability
make statistics
make figures
make paper
make artifact
```

Manifests include commit SHA, seed, device label, and `evidence_class`. Clean-checkout reproduction may earn internal reproduction only when harnesses pass; **independent reproduction is pending**.

## 20. Conclusion

Track B publication-grade work disables presence-bonus shortcuts, establishes semantic intervention causality via obs→act protocols, validates faithful DIAL and TarMAC end-to-end, and runs the largest scientifically useful final/ablation/generalization/robustness/intervention subset under CPU budgets—labeling `BLOCKED_COMPUTE_CAPACITY` when incomplete. Remaining blockers: full multi-scenario 5-seed matrix for every baseline, GPU measurements, physical pilots, DOI, and independent review.

## References

1. Foerster, J. et al. Learning to Communicate with Deep Multi-Agent Reinforcement Learning. NeurIPS 2016 (DIAL).
2. Das, A. et al. TarMAC: Targeted Multi-Agent Communication. ICML 2019.
3. Rashid, T. et al. QMIX: Monotonic Value Function Factorisation. ICML 2018.
4. Yu, C. et al. The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. NeurIPS 2022.
5. 3GPP TS 38.300 / NR overall description (system context only; no conformance claim).
