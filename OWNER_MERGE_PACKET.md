# OWNER_MERGE_PACKET — STREAM-C-PKT-001 C8

**Repo:** `gunnchos-emergent-service-intent-protocols`  
**Branch:** `cursor/oulu-publication-grade-science`  
**PR:** [#2](https://github.com/gunnchOS3k/gunnchos-emergent-service-intent-protocols/pull/2) — `OWNER_REVIEW`  
**Packet date (UTC):** 2026-08-16T16:26:28Z  
**Verifier host:** Edmunds-MacBook-Pro.local (Darwin arm64)

---

## SHAs

| Role | SHA |
|------|-----|
| Tip (`HEAD`) | `b07038e763d2a6923b98d3885a6707cb200281b9` |
| Base `origin/main` | `9b3fec9fdff752f97fbe9950b7db5d9874aea72d` |
| Merge-base | `9b3fec9fdff752f97fbe9950b7db5d9874aea72d` |

Branch commits ahead of `origin/main` (4):

1. `b8fbc82` — Disable message-presence reward bonuses; add semantic obs→act protocols.
2. `c5a69c1` — Make DIAL and TarMAC end-to-end faithful to their named algorithms.
3. `e1b9de8` — Add scientific tests for semantic interventions, DIAL, and TarMAC.
4. `b07038e` — Run publication-grade final suites and rewrite Results from actual outputs.

## Rebase status

**CLEAN.** `git fetch origin` then `HEAD..origin/main` empty; merge-base equals `origin/main`. No rebase required this pass. Branch already contains all commits vs `origin/main` @ `9b3fec9`.

## Test / experiment results summary

Primary gate run this packet: **`make gate4-cpu`** (exit 0).

| Suite | Command | Result |
|-------|---------|--------|
| Full pytest + coverage | `make test` → `pytest -q --cov=emergent_intent tests` | **PASS** — 116 collected / 116 passed; coverage **68.52%** (≥40% required) |
| Smoke | `make smoke` → `scripts/run_smoke_experiments.py --seeds 5 --steps 128` | **SUCCESS** — `SYNTHETIC_SIM` / `CPU_ONLY_APPLE_SILICON` |
| Causal | `make causal-tests` | **PASS** (21) |
| Algorithm validation | `make algorithm-validation` | **PASS** (22) |
| Semantic interventions | `make semantic-intervention-tests` | **PASS** (13) |
| DIAL validation | `make dial-validation` | **PASS** (8) |
| TarMAC validation | `make tarmac-validation` | **PASS** (8) |
| Gate banner | `GATE4_OULU_CPU_OK (smoke≠scientific final)` | printed |

Device / evidence (this host): torch `2.13.0`, CUDA unavailable → CPU-only Apple Silicon; evidence class **`SYNTHETIC_SIM`**.

### Prior final / companion suite artifacts (on branch under `results/`)

| Artifact | Status | Notes |
|----------|--------|-------|
| `results/final/STATUS.json` | `RAN` / `FINAL` / not blocked | 135 rows; 9 methods × 3 scenarios × 5 seeds @ 1024 steps; `SYNTHETIC_SIM` |
| `results/ablations/STATUS.json` | `PARTIAL` | 20 rows; subset of algorithms |
| `results/generalization/STATUS.json` | `RAN` | UE counts 1–3; digital-twin layouts still pending |
| `results/robustness/STATUS.json` | `RAN` | Partial (erasure_p); full domain-shift pending |
| `results/interventions/STATUS.json` | `RAN` | Semantic obs→action mapping evidence |

Smoke re-run artifacts from this verify were **not** committed (working tree restored).

## Independent-verify readiness checklist

- [ ] Fresh clone / clean checkout of tip `b07038e`
- [ ] `make bootstrap` (`pip install -e ".[dev]"`)
- [ ] `make gate4-cpu` → expect `GATE4_OULU_CPU_OK`
- [ ] Optional: `make test` alone → 116 passed
- [ ] Optional: inspect `results/*/STATUS.json` (final `RAN`, ablations `PARTIAL`)
- [ ] Optional: `make gate4-gpu` → expect honest `BLOCKED_HARDWARE` without CUDA
- [ ] Do **not** treat smoke as scientific final; do not invent SoA / deployment / OTA / GPU claims
- [ ] Owner (Edmund) sole merge authority — Cursor does not merge

## Merge-ready (Edmund / owner only)

**YES** — for owner merge of PR #2 as science/code package after owner review.

Rationale: rebase clean vs `origin/main`; `make gate4-cpu` green; mergeable on GitHub; no fixable test failures this pass. OPEN items below are publication / completeness gaps, not gate failures. Cursor did not and will not merge.

## OPEN items

1. Final matrix still missing **VDN** and **centralized_oracle** cells in this subset (baselines list incomplete vs full catalog).
2. Ablations marked **`PARTIAL`** (not full algorithm set).
3. Generalization UE sweep only **1–3**; higher UE counts (e.g. 12–16) and digital-twin layouts pending.
4. Robustness only **erasure_p**; full domain-shift pending.
5. **GPU / OTA / physical RF** evidence not claimed; `gate4-gpu` remains `BLOCKED_HARDWARE` on this host.
6. **DOI**, peer review, and **independent reproduction** still pending (README flags).
7. Manuscript remains draft-grade evidence (`SYNTHETIC_SIM`); no SoA / deployment superiority claims from this packet.

## Explicit non-action

**Cursor did not merge** PR #2. Merge is **OWNER_REVIEW** only (Edmund).
