from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from emergent_intent.comm.channel import SILENCE, ChannelConfig, MessageChannel
from emergent_intent.env.config import CommMode, EnvConfig, ScenarioFamily


class ServiceIntentEnv:
    """PettingZoo ParallelAPI-compatible partially observable multi-agent env."""

    metadata = {"name": "service_intent_v0", "render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.agents: list[str] = []
        self.possible_agents: list[str] = []
        self._t = 0
        self._obs_dim = 8
        self._act_dim = 4
        self._build_agents()
        self.channel = MessageChannel(
            ChannelConfig(
                mode=self.config.comm_mode.value,
                vocab_size=self.config.vocab_size,
                msg_len=self.config.msg_len,
                erasure_p=self.config.erasure_p,
                bit_error_p=self.config.bit_error_p,
                delay=self.config.delay,
            ),
            agents=self.possible_agents,
            rng=self.rng,
        )

    def _build_agents(self) -> None:
        agents = [f"ue_{i}" for i in range(self.config.n_ue)] + ["bs_0", "edge_0"]
        if self.config.include_ntn or self.config.scenario == ScenarioFamily.tn_ntn_failover:
            agents.append("ntn_relay")
        self.possible_agents = agents
        self.agents = agents[:]
        self.observation_spaces = {
            a: spaces.Box(low=-10, high=10, shape=(self._obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        # MultiDiscrete: [power_bins, sched_bins, msg_symbol_or_silence, boost]
        nvec = [5, 5, max(self.config.vocab_size + 1, 2), 3]
        self._nvec = nvec
        self.action_spaces = {
            a: spaces.MultiDiscrete(nvec)
            for a in self.possible_agents
        }

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.channel.rng = self.rng
        self.agents = self.possible_agents[:]
        self._t = 0
        n = self.config.n_ue
        self._state = {
            "queue": self.rng.uniform(0, 1, size=n),
            "aoi": self.rng.uniform(0, 5, size=n),
            "snr": self.rng.uniform(0.2, 1.0, size=n),
            "battery": self.rng.uniform(0.3, 1.0, size=n),
            "blockage": self.rng.uniform(0, 1, size=n),
            "fairness_debt": np.zeros(n),
            "tn_available": 1.0,
            "ntn_available": 1.0 if "ntn_relay" in self.agents else 0.0,
            "service_critical": np.array(
                [
                    1.0
                    if i == 0 and self.config.scenario == ScenarioFamily.critical_service
                    else 0.0
                    for i in range(n)
                ]
            ),
            "last_msgs": {},
            "bits_used": 0.0,
            "energy_used": 0.0,
            "deliveries": 0.0,
            "violations": 0.0,
        }
        if self.config.scenario == ScenarioFamily.terrestrial_congestion:
            self._state["queue"] = np.clip(self._state["queue"] * 1.5, 0, 5)
        if self.config.scenario == ScenarioFamily.tn_ntn_failover:
            self._state["tn_available"] = 0.2
            self._state["ntn_available"] = 0.9
        obs = {a: self._observe(a) for a in self.agents}
        infos = {a: {"partial_observability": True} for a in self.agents}
        return obs, infos

    def _observe(self, agent: str) -> np.ndarray:
        s = self._state
        noise = self.config.observation_noise

        def n(x: float) -> float:
            return float(x + self.rng.normal(0, noise)) if noise > 0 else float(x)

        if agent.startswith("ue_"):
            i = int(agent.split("_")[1])
            vec = np.array(
                [
                    n(s["queue"][i]),
                    n(s["aoi"][i]),
                    n(s["snr"][i]),
                    n(s["battery"][i]),
                    n(s["blockage"][i]),
                    n(s["fairness_debt"][i]),
                    0.0,
                    float(self._t) / self.config.horizon,
                ],
                dtype=np.float32,
            )
        elif agent.startswith("bs_"):
            vec = np.array(
                [
                    n(s["queue"].mean()),
                    n(s["aoi"].mean()),
                    n(s["snr"].mean()),
                    n(s["tn_available"]),
                    n(s["ntn_available"]),
                    n(s["bits_used"] / max(self.config.message_bit_budget, 1)),
                    0.0,
                    float(self._t) / self.config.horizon,
                ],
                dtype=np.float32,
            )
        elif agent.startswith("edge_"):
            vec = np.array(
                [
                    n(s["deliveries"]),
                    n(s["energy_used"]),
                    n(s["violations"]),
                    n(s["fairness_debt"].mean()),
                    n(s["queue"].max()),
                    1.0 if self.config.scenario == ScenarioFamily.education_fairness else 0.0,
                    0.0,
                    float(self._t) / self.config.horizon,
                ],
                dtype=np.float32,
            )
        else:
            vec = np.array(
                [
                    n(s["ntn_available"]),
                    n(1.0 - s["tn_available"]),
                    n(s["aoi"].mean()),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float(self._t) / self.config.horizon,
                ],
                dtype=np.float32,
            )
        return vec

    def _fixed_protocol_message(self, agent: str) -> np.ndarray | None:
        if agent.startswith("ue_"):
            i = int(agent.split("_")[1])
            q = self._state["queue"][i]
            vals = [int(q > 0.5), int(self._state["aoi"][i] > 2)]
            return np.array(vals[: self.config.msg_len])
        return None

    def step(self, actions: dict[str, np.ndarray]):
        s = self._state
        bits = 0.0
        energy = 0.0
        deliveries = 0.0

        for agent, act in actions.items():
            act = np.asarray(act).ravel()
            # Support Box floats or MultiDiscrete ints
            if act.dtype.kind == "f":
                power = float(np.clip(act[0], 0, self.config.max_power))
                msg_token = float(act[2]) if act.size > 2 else 0.0
                continuous = act
            else:
                power = float(act[0]) / max(self._nvec[0] - 1, 1) * self.config.max_power
                msg_token = float(act[2])
                continuous = act.astype(np.float64) / np.maximum(np.asarray(self._nvec) - 1, 1)
            energy += power * 0.1
            msg = None
            if self.config.comm_mode == CommMode.no_comm:
                msg = None
            elif self.config.comm_mode == CommMode.fixed_protocol:
                msg = self._fixed_protocol_message(agent)
            elif self.config.comm_mode == CommMode.discrete_learned:
                if msg_token <= 0:
                    msg = np.array([SILENCE])
                else:
                    sym = int(msg_token - 1) % self.config.vocab_size
                    msg = np.array([sym] * self.config.msg_len)
            else:
                msg = continuous

            if self.config.comm_mode == CommMode.continuous_learned:
                recv = self.channel.send_continuous(msg if isinstance(msg, np.ndarray) else None)
            else:
                recv = self.channel.send_discrete(msg if isinstance(msg, np.ndarray) else None)
                bits += self.channel.bit_cost(msg if isinstance(msg, np.ndarray) else None)
            s["last_msgs"][agent] = recv

        n = self.config.n_ue
        arrivals = self.rng.uniform(0, 0.3, size=n)
        s["queue"] = np.clip(s["queue"] + arrivals - 0.2 * s["snr"] * (1 - s["blockage"]), 0, 5)
        s["aoi"] = s["aoi"] + 1.0
        served = (s["queue"] < 0.5).astype(float)
        deliveries += float(served.sum())
        s["aoi"] = np.where(served > 0, 0.0, s["aoi"])
        s["fairness_debt"] += served.max() - served

        violations = 0.0
        if bits > self.config.message_bit_budget:
            violations += 1.0
        if energy > self.config.max_power * n:
            violations += 1.0
        jain = (
            (served.sum() ** 2) / (n * (served**2).sum() + 1e-8) if served.sum() > 0 else 0.0
        )
        if jain < self.config.fairness_floor and self.config.scenario == ScenarioFamily.education_fairness:
            violations += 1.0
        if (
            self.config.scenario == ScenarioFamily.critical_service
            and s["service_critical"][0] > 0
            and served[0] < 1
        ):
            violations += 1.0

        s["bits_used"] += bits
        s["energy_used"] += energy
        s["deliveries"] += deliveries
        s["violations"] += violations
        self._t += 1

        reward_vec = {
            "task_success": deliveries / max(n, 1),
            "latency_aoi": -float(s["aoi"].mean()) / 10.0,
            "energy": -energy,
            "message_bits": -bits / max(self.config.message_bit_budget, 1),
            "fairness": float(jain),
            "spectral_efficiency": float(deliveries / (energy + 1e-6)),
            "violations": -violations,
        }
        wts = {
            "task_success": 1.0,
            "latency_aoi": 0.2,
            "energy": 0.1,
            "message_bits": 0.1,
            "fairness": 0.2,
            "spectral_efficiency": 0.05,
            "violations": 1.0,
        }
        scalar = sum(wts[k] * reward_vec[k] for k in wts)
        rewards = {a: float(scalar) for a in self.agents}
        term = self._t >= self.config.horizon
        terminations = {a: term for a in self.agents}
        truncations = {a: False for a in self.agents}
        obs = {a: self._observe(a) for a in self.agents}
        infos = {
            a: {
                "reward_vec": reward_vec,
                "bits": bits,
                "evidence_label": "SYNTHETIC_EXPERIMENT",
            }
            for a in self.agents
        }
        if term:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def episode_summary(self) -> dict[str, float]:
        s = self._state
        return {
            "mean_task_success": float(s.get("deliveries", 0.0) / max(self._t, 1)),
            "mean_energy": float(s.get("energy_used", 0.0) / max(self._t, 1)),
            "mean_message_bits": float(s.get("bits_used", 0.0) / max(self._t, 1)),
            "mean_violations": float(s.get("violations", 0.0) / max(self._t, 1)),
            "steps": float(self._t),
        }

    def state(self) -> dict:
        """Centralized oracle access — upper-bound baseline only."""
        return dict(self._state)


AIRANDocPOMDP = ServiceIntentEnv


def make_env(config: EnvConfig | None = None, **kwargs) -> ServiceIntentEnv:
    if config is None:
        config = EnvConfig(**kwargs) if kwargs else EnvConfig()
    elif kwargs:
        config = config.model_copy(update=kwargs)
    return ServiceIntentEnv(config)
