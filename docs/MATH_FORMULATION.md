# Mathematical formulation (Doc-POMDP)

Notation matches `src/emergent_intent/env/wireless_env.py` and `src/emergent_intent/comm/channel.py`.

## Agents and time

Let $\mathcal{I}$ be the agent set:
$\mathcal{I} = \{\mathrm{ue\_0}, \mathrm{bs\_0}, \mathrm{orchestrator}\} \cup \{\mathrm{ntn\_relay}\}$ (optional).

Discrete time $t = 0,\ldots,T-1$ with horizon $T$ (`EnvConfig.horizon`).

## Global state

Global state $s_t \in \mathcal{S}$ (not fully observed) includes:
$$
s_t = (\ell_t, q_t, d_t, e_t, f_t, n_t, u_t, \tau_t)
$$
load $\ell$, link quality $q$, queue delay $d$ (ms), residual energy $e$, fairness gap $f$, NTN availability $n$, spectrum util $u$, TN link $\tau$.

## Observations (partial)

Agent $i$ receives
$$
o^i_t = M_i(s_t) + \varepsilon^i_t \,\|\, m^{i,\mathrm{in}}_t
$$
where $M_i$ is a binary mask (`_observe`), $\varepsilon$ is i.i.d. Gaussian noise (`observation_noise`), and $m^{i,\mathrm{in}}$ is the inbox from the message channel.

## Actions and messages

Control action $a^i_t$ is MultiDiscrete. Message $m^{i,\mathrm{out}}_t$ depends on channel mode:
- `no_comm`: empty
- `fixed_protocol`: hand-designed symbols
- `continuous_learned`: $m \in \mathbb{R}^{d}$
- `discrete_learned`: $m \in \{0,\ldots,V\}^L$ with optional silence id $V$

Channel impairments: erasure probability $p_e$, corruption $p_c$, delay $\delta$ steps, bit cost $c_b$.

Targeted delivery (optional): adjacency / attention restricts receivers (TarMAC-style).

## Transition and rewards

Synthetic dynamics $s_{t+1} \sim P(\cdot \mid s_t, a_t, m_t)$ update congestion, TN–NTN failover, energy, fairness.

Per-step metric vector
$$
\mathbf{r}_t = (r^{\mathrm{task}}, r^{\mathrm{lat}}, r^{\mathrm{en}}, r^{\mathrm{bits}}, r^{\mathrm{fair}}, r^{\mathrm{se}}, r^{\mathrm{viol}})
$$
Scalarization (default):
$$
R_t = \sum_k w_k \, u_k(\mathbf{r}_t)
$$
with utilities $u_k$ mapping to higher-is-better (`normalize_metrics`). Optional preference conditioning $\omega$ or Lagrangian penalties $\lambda^\top g(s,a)$.

## Learning objectives

- **IPPO**: independent actors/critics on local $o^i$
- **MAPPO**: centralized critic on $s$ (oracle `state()`), decentralized actors
- **VDN / QMIX**: value factorization over team reward
- **DIAL/TarMAC**: Gumbel-Softmax message training + hard discrete execution + attention targeting

## Evidence class

All default simulations are labeled `SYNTHETIC_SIM`. Hardware measurements require separate labeled profiles and must not be mixed into synthetic tables.
