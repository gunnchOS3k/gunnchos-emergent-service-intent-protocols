from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from emergent_intent.comm.channel import (
    SILENCE,
    ChannelConfig,
    MessageChannel,
    slot_dim,
)
from emergent_intent.env.config import CommMode, EnvConfig, ScenarioFamily

# Action factor indices
A_POWER = 0
A_PRB = 1
A_MCS = 2
A_ACCESS = 3  # 0=TN, 1=NTN, 2=auto, 3=deny
A_HANDOVER = 4
A_ROUTING = 5
A_ADMISSION = 6
A_PRIORITY = 7
A_OFFLOAD = 8
A_MSG = 9
A_TARGET = 10

LOCAL_OBS_DIM = 12


class ServiceIntentEnv:
    """PettingZoo ParallelAPI Doc-POMDP with causal radio control and message inboxes."""

    metadata = {"name": "service_intent_v0", "render_modes": []}

    # Action name mapping for causal tests / docs
    ACTION_NAMES = (
        "power",
        "prb",
        "mcs",
        "access",
        "handover",
        "routing",
        "admission",
        "priority",
        "offload",
        "message",
        "target",
    )

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.agents: list[str] = []
        self.possible_agents: list[str] = []
        self._t = 0
        self._build_agents()
        self._msg_payload_len = (
            self.config.continuous_dim
            if self.config.comm_mode == CommMode.continuous_learned
            else self.config.msg_len
        )
        self._inbox_dim = self.config.inbox_capacity * slot_dim(self._msg_payload_len)
        self._obs_dim = LOCAL_OBS_DIM + self._inbox_dim
        n_agents = len(self.possible_agents)
        # MultiDiscrete action factors (§6.2)
        self._nvec = [
            5,  # power
            5,  # PRB / resource blocks
            4,  # MCS
            4,  # access selection
            3,  # handover intensity
            3,  # routing
            2,  # admission
            3,  # priority
            2,  # offload
            max(self.config.vocab_size + 1, 2),  # message token (0=silence)
            max(n_agents, 1),  # target index
        ]
        self.observation_spaces = {
            a: spaces.Box(low=-10, high=10, shape=(self._obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {a: spaces.MultiDiscrete(self._nvec) for a in self.possible_agents}
        self.channel = MessageChannel(
            ChannelConfig(
                mode=self.config.comm_mode.value,
                vocab_size=self.config.vocab_size,
                msg_len=self.config.msg_len,
                continuous_dim=self.config.continuous_dim,
                erasure_p=self.config.erasure_p,
                bit_error_p=self.config.bit_error_p,
                corruption_p=self.config.corruption_p,
                delay=self.config.delay,
                targeted=self.config.targeted,
                loopback=self.config.loopback,
                inbox_capacity=self.config.inbox_capacity,
                stale_threshold=self.config.stale_threshold,
                bit_cost=self.config.bit_cost,
            ),
            agents=self.possible_agents,
            rng=self.rng,
        )
        self._last_inbox: dict[str, list] = {}
        # Presence bonuses disabled: messages help only via observations→actions.
        self._coordination_bonus = 0.0
        self.message_presence_bonus_enabled = False

    def _build_agents(self) -> None:
        agents = [f"ue_{i}" for i in range(self.config.n_ue)] + ["bs_0", "edge_0"]
        if self.config.include_ntn or self.config.scenario in (
            ScenarioFamily.tn_ntn_failover,
            ScenarioFamily.tn_ntn_continuity,
        ):
            agents.append("ntn_relay")
        self.possible_agents = agents
        self.agents = agents[:]

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
        self.channel.reset(self.rng)
        self._last_inbox = {a: [] for a in self.agents}
        self._coordination_bonus = 0.0
        n = self.config.n_ue
        self._state = {
            "queue": self.rng.uniform(0.5, 2.0, size=n),
            "aoi": self.rng.uniform(0, 3, size=n),
            "snr": self.rng.uniform(0.3, 1.0, size=n),
            "battery": self.rng.uniform(0.4, 1.0, size=n),
            "blockage": self.rng.uniform(0, 1, size=n),
            "congestion": float(self.rng.uniform(0.2, 0.8)),
            "fairness_debt": np.zeros(n),
            "tn_available": 1.0,
            "ntn_available": 1.0 if "ntn_relay" in self.agents else 0.0,
            "ntn_cost": 0.3,
            "service_critical": np.array(
                [
                    1.0
                    if i == 0
                    and self.config.scenario
                    in (ScenarioFamily.critical_service, ScenarioFamily.hidden_blockage_congestion)
                    else 0.0
                    for i in range(n)
                ]
            ),
            "service_intent": 1.0
            if self.config.scenario
            in (ScenarioFamily.tn_ntn_continuity, ScenarioFamily.critical_service)
            else 0.5,
            "access_mode": np.zeros(n),  # 0 TN, 1 NTN
            "served_bits": np.zeros(n),
            "bits_used": 0.0,
            "energy_used": 0.0,
            "deliveries": 0.0,
            "violations": 0.0,
            "interference": 0.1,
            "last_actions": {},
        }
        self._apply_scenario_init()
        obs = {a: self._observe(a) for a in self.agents}
        infos = {
            a: {
                "partial_observability": True,
                "scenario": self.config.scenario.value,
                "comm_necessity": self._comm_necessity_note(),
            }
            for a in self.agents
        }
        return obs, infos

    def _apply_scenario_init(self) -> None:
        s = self._state
        sc = self.config.scenario
        if sc == ScenarioFamily.terrestrial_congestion:
            s["congestion"] = 0.85
            s["queue"] = np.clip(s["queue"] * 1.4, 0, 5)
        elif sc == ScenarioFamily.tn_ntn_failover:
            s["tn_available"] = 0.15
            s["ntn_available"] = 0.95
        elif sc == ScenarioFamily.tn_ntn_continuity:
            s["tn_available"] = 0.1
            s["ntn_available"] = 0.9
            s["ntn_cost"] = 0.25
            s["snr"] = s["snr"] * 0.5
        elif sc == ScenarioFamily.hidden_blockage_congestion:
            # UE 0 heavily blocked; BS sees congestion but not full blockage vector
            s["blockage"][0] = 0.95
            if n := self.config.n_ue:
                if n > 1:
                    s["blockage"][1:] = self.rng.uniform(0.1, 0.4, size=n - 1)
            s["congestion"] = 0.9
            s["service_critical"][0] = 1.0
        elif sc == ScenarioFamily.critical_service:
            s["service_critical"][0] = 1.0
            s["queue"][0] = max(float(s["queue"][0]), 1.5)
        elif sc == ScenarioFamily.education_fairness:
            s["fairness_debt"] = self.rng.uniform(0.2, 0.8, size=self.config.n_ue)

    def _comm_necessity_note(self) -> str:
        sc = self.config.scenario
        if sc == ScenarioFamily.hidden_blockage_congestion:
            return (
                "UE sees local blockage; BS sees congestion not full UE blockage; "
                "edge sees priority — resource assignment needs messaging."
            )
        if sc in (ScenarioFamily.tn_ntn_continuity, ScenarioFamily.tn_ntn_failover):
            return (
                "UE sees TN degradation; BS sees TN load; NTN sees relay cost/availability; "
                "orchestrator sees intent — failover needs distributed communication."
            )
        return "Communication optional for coordination gains."

    def _noise(self, x: float) -> float:
        noise = self.config.observation_noise
        return float(x + self.rng.normal(0, noise)) if noise > 0 else float(x)

    def _local_features(self, agent: str) -> np.ndarray:
        """Partial local observation — asymmetric by design for comm-necessary scenarios."""
        s = self._state
        n = self.config.n_ue
        tnorm = float(self._t) / max(self.config.horizon, 1)
        sc = self.config.scenario
        hidden = sc == ScenarioFamily.hidden_blockage_congestion
        continuity = sc in (ScenarioFamily.tn_ntn_continuity, ScenarioFamily.tn_ntn_failover)

        if agent.startswith("ue_"):
            i = int(agent.split("_")[1])
            # UE sees local blockage/snr/queue; not global congestion fully
            return np.array(
                [
                    self._noise(s["queue"][i]),
                    self._noise(s["aoi"][i]),
                    self._noise(s["snr"][i]),
                    self._noise(s["battery"][i]),
                    self._noise(s["blockage"][i]),
                    self._noise(s["fairness_debt"][i]),
                    self._noise(s["access_mode"][i]),
                    self._noise(0.0 if hidden else s["congestion"]),  # hidden from UE? actually UE doesn't need it
                    self._noise(s["tn_available"] if continuity else 0.0),
                    self._noise(0.0),  # no ntn cost locally unless continuity tip
                    self._noise(s["service_critical"][i]),
                    tnorm,
                ],
                dtype=np.float32,
            )
        if agent.startswith("bs_"):
            # BS sees congestion / average queue / TN avail; NOT each UE's full blockage in Scenario A
            blk = 0.0 if hidden else float(s["blockage"].mean())
            return np.array(
                [
                    self._noise(s["queue"].mean()),
                    self._noise(s["aoi"].mean()),
                    self._noise(s["snr"].mean()),
                    self._noise(s["congestion"]),
                    self._noise(s["tn_available"]),
                    self._noise(s["ntn_available"]),
                    self._noise(s["bits_used"] / max(self.config.message_bit_budget, 1)),
                    self._noise(blk),
                    self._noise(s["interference"]),
                    self._noise(0.0),  # no per-UE critical vector
                    self._noise(0.0),
                    tnorm,
                ],
                dtype=np.float32,
            )
        if agent.startswith("edge_"):
            # Edge sees priority / intent / violations; not full radio state
            return np.array(
                [
                    self._noise(s["deliveries"]),
                    self._noise(s["energy_used"]),
                    self._noise(s["violations"]),
                    self._noise(s["fairness_debt"].mean()),
                    self._noise(s["queue"].max()),
                    self._noise(s["service_intent"]),
                    self._noise(float(s["service_critical"].sum())),
                    self._noise(1.0 if sc == ScenarioFamily.education_fairness else 0.0),
                    self._noise(0.0 if hidden else s["congestion"]),
                    self._noise(0.0),
                    self._noise(0.0),
                    tnorm,
                ],
                dtype=np.float32,
            )
        # ntn_relay
        return np.array(
            [
                self._noise(s["ntn_available"]),
                self._noise(1.0 - s["tn_available"]),
                self._noise(s["ntn_cost"]),
                self._noise(s["aoi"].mean()),
                self._noise(s["snr"].mean() * 0.5),
                self._noise(0.0),
                self._noise(0.0),
                self._noise(0.0),
                self._noise(0.0),
                self._noise(0.0),
                self._noise(0.0),
                tnorm,
            ],
            dtype=np.float32,
        )

    def _observe(self, agent: str) -> np.ndarray:
        local = self._local_features(agent)
        if self.config.comm_mode == CommMode.no_comm:
            inbox = np.zeros(self._inbox_dim, dtype=np.float32)
        else:
            inbox = self.channel.inbox_vector(agent, n_agents=len(self.possible_agents))
        return np.concatenate([local, inbox]).astype(np.float32)

    def _decode_action(self, act: np.ndarray) -> dict[str, float | int | None]:
        act = np.asarray(act).ravel()
        # Pad short actions for backward compatibility
        if act.size < len(self._nvec):
            pad = np.zeros(len(self._nvec) - act.size)
            act = np.concatenate([act, pad])
        if act.dtype.kind == "f":
            # continuous Box-style: scale from [0,1] fractions if needed
            vals = act.astype(np.float64)
            power = float(np.clip(vals[A_POWER], 0, self.config.max_power))
            prb = float(np.clip(vals[A_PRB], 0, 1))
            mcs = float(np.clip(vals[A_MCS], 0, 1))
            access = int(np.clip(vals[A_ACCESS], 0, 3))
            handover = float(np.clip(vals[A_HANDOVER], 0, 1))
            routing = float(np.clip(vals[A_ROUTING], 0, 1))
            admission = int(vals[A_ADMISSION] > 0.5)
            priority = float(np.clip(vals[A_PRIORITY], 0, 1))
            offload = float(np.clip(vals[A_OFFLOAD], 0, 1))
            msg_token = float(vals[A_MSG])
            target_i = int(np.clip(vals[A_TARGET], 0, len(self.possible_agents) - 1))
        else:
            power = float(act[A_POWER]) / max(self._nvec[A_POWER] - 1, 1) * self.config.max_power
            prb = float(act[A_PRB]) / max(self._nvec[A_PRB] - 1, 1)
            mcs = float(act[A_MCS]) / max(self._nvec[A_MCS] - 1, 1)
            access = int(act[A_ACCESS])
            handover = float(act[A_HANDOVER]) / max(self._nvec[A_HANDOVER] - 1, 1)
            routing = float(act[A_ROUTING]) / max(self._nvec[A_ROUTING] - 1, 1)
            admission = int(act[A_ADMISSION])
            priority = float(act[A_PRIORITY]) / max(self._nvec[A_PRIORITY] - 1, 1)
            offload = float(act[A_OFFLOAD]) / max(self._nvec[A_OFFLOAD] - 1, 1)
            msg_token = float(act[A_MSG])
            target_i = int(act[A_TARGET]) % len(self.possible_agents)
        return {
            "power": power,
            "prb": prb,
            "mcs": mcs,
            "access": access,
            "handover": handover,
            "routing": routing,
            "admission": admission,
            "priority": priority,
            "offload": offload,
            "msg_token": msg_token,
            "target_i": target_i,
        }

    def _fixed_protocol_message(self, agent: str) -> np.ndarray:
        from emergent_intent.comm.semantic_protocol import encode_fixed_protocol_message

        return encode_fixed_protocol_message(agent, self._state, msg_len=self.config.msg_len)

    def _build_outbound(
        self, decoded: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, np.ndarray], dict[str, str | None]]:
        outbound: dict[str, np.ndarray] = {}
        targets: dict[str, str | None] = {}
        for agent, d in decoded.items():
            msg = None
            if self.config.comm_mode == CommMode.no_comm:
                msg = self.channel.empty_message()
            elif self.config.comm_mode == CommMode.fixed_protocol:
                msg = self._fixed_protocol_message(agent)
            elif self.config.comm_mode == CommMode.discrete_learned:
                tok = d["msg_token"]
                if tok <= 0:
                    msg = np.full(self.config.msg_len, self.channel.silence_id(), dtype=np.float32)
                else:
                    sym = int(tok - 1) % self.config.vocab_size
                    msg = np.full(self.config.msg_len, sym, dtype=np.float32)
            else:
                # continuous: encode token into a small vector
                tok = float(d["msg_token"])
                msg = np.full(self.config.continuous_dim, tok / max(self._nvec[A_MSG], 1), dtype=np.float32)
            outbound[agent] = msg
            tgt_i = int(d["target_i"])
            targets[agent] = self.possible_agents[tgt_i] if self.config.targeted else None
        return outbound, targets

    def _message_coordination(self, inbox_map: dict[str, list]) -> float:
        """Legacy presence bonus — DISABLED for publication-grade science.

        Returns 0.0 unless ``message_presence_bonus_enabled`` is explicitly set.
        Messages must influence outcomes only through observations→actions.
        """
        if not getattr(self, "message_presence_bonus_enabled", False):
            return 0.0
        # Kept for ablation comparison only (never default-on).
        sc = self.config.scenario
        bonus = 0.0
        if sc == ScenarioFamily.hidden_blockage_congestion:
            bs_slots = inbox_map.get("bs_0", [])
            edge_slots = inbox_map.get("edge_0", [])
            ue_to_bs = any(r.sender_id == 0 and r.valid > 0 for r in bs_slots)
            edge_prio = any(r.valid > 0 for r in edge_slots)
            if ue_to_bs:
                bonus += 0.35
            if edge_prio and ue_to_bs:
                bonus += 0.25
        elif sc in (ScenarioFamily.tn_ntn_continuity, ScenarioFamily.tn_ntn_failover):
            for key in ("ntn_relay", "bs_0", "edge_0"):
                if any(r.valid > 0 for r in inbox_map.get(key, [])):
                    bonus += 0.15
        return bonus

    def _service_probability(
        self,
        ue_i: int,
        ue_act: dict[str, Any],
        bs_act: dict[str, Any],
        edge_act: dict[str, Any],
        ntn_act: dict[str, Any] | None,
        coord: float,
    ) -> float:
        """Causal service model: depends on power, PRB, MCS, access, congestion, etc."""
        s = self._state
        snr = float(s["snr"][ue_i])
        blockage = float(s["blockage"][ue_i])
        queue = float(s["queue"][ue_i])
        cong = float(s["congestion"])
        interf = float(s["interference"])

        power = float(ue_act["power"])
        prb = float(bs_act["prb"])
        mcs = float(bs_act["mcs"])
        admission = int(bs_act["admission"])
        priority = max(float(bs_act["priority"]), float(edge_act["priority"]))
        routing = float(bs_act["routing"])
        offload = float(edge_act["offload"])
        handover = float(ue_act["handover"])

        # Access / handover selection
        access = int(ue_act["access"])
        if access == 3 or admission == 0:
            return 0.0  # denied / not admitted
        use_ntn = False
        if access == 1:
            use_ntn = True
        elif access == 2:  # auto
            use_ntn = s["tn_available"] < 0.4 or handover > 0.5
        else:
            use_ntn = False
        # Wrong handover: force NTN when unavailable, or TN when TN dead
        if ntn_act is not None and float(ntn_act.get("handover", 0)) > 0.6:
            use_ntn = True
        s["access_mode"][ue_i] = 1.0 if use_ntn else 0.0

        if use_ntn:
            link_q = float(s["ntn_available"]) * (1.0 - 0.5 * float(s["ntn_cost"]))
            if ntn_act is not None:
                link_q *= 0.5 + 0.5 * float(ntn_act["power"])
            # wrong: choosing NTN when unavailable
            if s["ntn_available"] < 0.2:
                link_q *= 0.1
        else:
            link_q = float(s["tn_available"]) * (1.0 - blockage) * snr
            if s["tn_available"] < 0.2:
                link_q *= 0.15

        # Resource / MCS / power causality
        p_svc = (
            0.05
            + 0.35 * power
            + 0.30 * prb
            + 0.15 * mcs
            + 0.20 * link_q
            - 0.25 * cong
            - 0.15 * interf
            + 0.10 * priority * float(s["service_critical"][ue_i])
            + 0.08 * routing
            + 0.05 * offload
            + 0.20 * coord
        )
        # Queue pressure: higher queue needs service but congestion hurts
        p_svc *= 1.0 / (1.0 + 0.15 * max(queue - 1.0, 0.0))
        # Critical priority boost
        if s["service_critical"][ue_i] > 0 and priority > 0.5:
            p_svc += 0.25
        elif s["service_critical"][ue_i] > 0 and priority < 0.2:
            p_svc -= 0.2
        return float(np.clip(p_svc, 0.0, 0.98))

    def step(self, actions: dict[str, np.ndarray]):
        s = self._state
        decoded = {a: self._decode_action(act) for a, act in actions.items()}
        s["last_actions"] = decoded

        outbound, targets = self._build_outbound(decoded)
        flat_inbox, bits, inbox_map = self.channel.exchange(outbound, self.rng, targets=targets)
        self._last_inbox = inbox_map
        coord = self._message_coordination(inbox_map)
        self._coordination_bonus = coord

        n = self.config.n_ue
        bs_act = decoded.get("bs_0", self._decode_action(np.zeros(len(self._nvec))))
        edge_act = decoded.get("edge_0", self._decode_action(np.zeros(len(self._nvec))))
        ntn_act = decoded.get("ntn_relay") if "ntn_relay" in decoded else None

        energy = 0.0
        for a, d in decoded.items():
            energy += float(d["power"]) * 0.1
            if a.startswith("ue_"):
                i = int(a.split("_")[1])
                s["battery"][i] = max(0.0, s["battery"][i] - 0.02 * float(d["power"]))

        # Interference rises with aggregate power
        energy_ue = sum(float(decoded[f"ue_{i}"]["power"]) for i in range(n) if f"ue_{i}" in decoded)
        s["interference"] = 0.05 + 0.15 * energy_ue / max(n, 1)

        arrivals = self.rng.uniform(0.05, 0.35, size=n)
        served = np.zeros(n)
        served_bits = np.zeros(n)
        for i in range(n):
            ue_key = f"ue_{i}"
            ue_act = decoded.get(ue_key, self._decode_action(np.zeros(len(self._nvec))))
            p = self._service_probability(i, ue_act, bs_act, edge_act, ntn_act, coord)
            # Stochastic service with causal expectation
            if self.rng.random() < p:
                # Served bits scale with PRB * MCS * power
                bits_i = float(bs_act["prb"]) * (0.5 + float(bs_act["mcs"])) * (0.5 + float(ue_act["power"]))
                drain = min(float(s["queue"][i]), 0.3 + bits_i)
                s["queue"][i] = max(0.0, float(s["queue"][i]) - drain)
                served[i] = 1.0
                served_bits[i] = drain
            s["queue"][i] = min(5.0, float(s["queue"][i]) + arrivals[i])

        s["aoi"] = s["aoi"] + 1.0
        s["aoi"] = np.where(served > 0, 0.0, s["aoi"])
        s["served_bits"] += served_bits
        s["fairness_debt"] += served.max() - served if served.sum() > 0 else 0.0

        deliveries = float(served.sum())
        violations = 0.0
        if bits > self.config.message_bit_budget:
            violations += 1.0
        if energy > self.config.max_power * (n + 2):
            violations += 1.0
        jain = (
            (served.sum() ** 2) / (n * (served**2).sum() + 1e-8) if served.sum() > 0 else 0.0
        )
        if jain < self.config.fairness_floor and self.config.scenario == ScenarioFamily.education_fairness:
            violations += 1.0
        if (
            self.config.scenario
            in (ScenarioFamily.critical_service, ScenarioFamily.hidden_blockage_congestion)
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
            "spectral_efficiency": float(
                np.clip(served_bits.sum() / (energy + 1e-3), 0.0, 20.0)
            ),
            "violations": -violations,
        }
        wts = self.config.objectives or {
            "task_success": 1.0,
            "latency_aoi": 0.2,
            "energy": 0.1,
            "message_bits": 0.1,
            "fairness": 0.2,
            "spectral_efficiency": 0.05,
            "violations": 1.0,
        }
        scalar = sum(wts.get(k, 0.0) * reward_vec[k] for k in reward_vec)
        rewards = {a: float(scalar) for a in self.agents}
        term = self._t >= self.config.horizon
        terminations = {a: term for a in self.agents}
        truncations = {a: False for a in self.agents}
        obs = {a: self._observe(a) for a in self.agents}
        infos = {
            a: {
                "reward_vec": reward_vec,
                "bits": bits,
                "coordination": coord,
                "served": served.tolist(),
                "served_bits": served_bits.tolist(),
                "inbox_slots": len(inbox_map.get(a, [])),
                "evidence_label": "SYNTHETIC_EXPERIMENT",
                "evidence_class": self.config.evidence_class,
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
            "mean_served_bits": float(np.sum(s.get("served_bits", 0.0)) / max(self._t, 1)),
            "steps": float(self._t),
        }

    def state(self) -> dict:
        """Centralized oracle access — upper-bound baseline only."""
        return dict(self._state)

    def oracle_service_actions(self) -> dict[str, np.ndarray]:
        """Near-optimal coordinated actions with global state (upper bound)."""
        actions = {}
        nvec = self._nvec
        for a in self.possible_agents:
            act = np.zeros(len(nvec), dtype=np.int64)
            act[A_POWER] = nvec[A_POWER] - 1
            act[A_PRB] = nvec[A_PRB] - 1
            act[A_MCS] = nvec[A_MCS] - 1
            act[A_ADMISSION] = 1
            act[A_PRIORITY] = nvec[A_PRIORITY] - 1
            act[A_ROUTING] = nvec[A_ROUTING] - 1
            if self._state["tn_available"] < 0.4 and "ntn_relay" in self.possible_agents:
                act[A_ACCESS] = 1  # NTN
                act[A_HANDOVER] = nvec[A_HANDOVER] - 1
            else:
                act[A_ACCESS] = 0
            if a.startswith("ue_"):
                i = int(a.split("_")[1])
                # Message blockage to BS under fixed/discrete
                act[A_MSG] = 1 + int(self._state["blockage"][i] > 0.5)
                act[A_TARGET] = self.possible_agents.index("bs_0")
            elif a == "bs_0":
                act[A_MSG] = 1 + int(self._state["congestion"] > 0.5)
                act[A_TARGET] = self.possible_agents.index("edge_0")
            elif a == "edge_0":
                act[A_MSG] = 1 + int(self._state["service_intent"] > 0.5)
                act[A_OFFLOAD] = 1
                act[A_TARGET] = self.possible_agents.index("bs_0")
            elif a == "ntn_relay":
                act[A_MSG] = 1
                act[A_TARGET] = self.possible_agents.index("bs_0")
            actions[a] = act
        return actions


AIRANDocPOMDP = ServiceIntentEnv


def make_env(config: EnvConfig | None = None, **kwargs) -> ServiceIntentEnv:
    if config is None:
        config = EnvConfig(**kwargs) if kwargs else EnvConfig()
    elif kwargs:
        config = config.model_copy(update=kwargs)
    return ServiceIntentEnv(config)
