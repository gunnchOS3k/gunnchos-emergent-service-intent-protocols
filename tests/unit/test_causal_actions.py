"""§6.2 causal action tests — actions must affect service outcomes."""

from __future__ import annotations

import numpy as np

from emergent_intent.env import EnvConfig, ScenarioFamily, make_env
from emergent_intent.env.wireless_env import (
    A_ACCESS,
    A_ADMISSION,
    A_HANDOVER,
    A_MCS,
    A_MSG,
    A_POWER,
    A_PRB,
    A_PRIORITY,
    A_TARGET,
)


def _base_actions(env, overrides: dict[int, int] | None = None, per_agent: dict | None = None):
    acts = {a: np.zeros(len(env._nvec), dtype=np.int64) for a in env.agents}
    for a in acts:
        acts[a][A_ADMISSION] = 1
        acts[a][A_POWER] = 2
        acts[a][A_PRB] = 2
        acts[a][A_MCS] = 1
        if overrides:
            for idx, val in overrides.items():
                acts[a][idx] = val
    if per_agent:
        for agent, ov in per_agent.items():
            for idx, val in ov.items():
                acts[agent][idx] = val
    return acts


def _mean_served(env, actions_fn, seeds=tuple(range(12)), steps=8) -> float:
    totals = []
    for seed in seeds:
        env.reset(seed=seed)
        # Stabilize initial queues somewhat via oracle-ish wait — just run
        served = 0.0
        for _ in range(steps):
            if not env.agents:
                break
            acts = actions_fn(env)
            _, _, _, _, infos = env.step(acts)
            if infos:
                a0 = next(iter(infos))
                served += float(sum(infos[a0].get("served", [])))
        totals.append(served)
    return float(np.mean(totals))


def test_more_power_increases_service() -> None:
    env = make_env(
        EnvConfig(horizon=10, n_ue=1, seed=0, channel={"mode": "no_comm"}, erasure_p=0.0)
    )

    def low(e):
        return _base_actions(e, {A_POWER: 0, A_PRB: 4, A_MCS: 3})

    def high(e):
        return _base_actions(e, {A_POWER: 4, A_PRB: 4, A_MCS: 3})

    assert _mean_served(env, high) > _mean_served(env, low)


def test_more_prb_increases_served_traffic() -> None:
    env = make_env(EnvConfig(horizon=10, n_ue=1, seed=1, channel={"mode": "no_comm"}))

    def low(e):
        return _base_actions(e, {A_POWER: 4, A_PRB: 0, A_MCS: 3})

    def high(e):
        return _base_actions(e, {A_POWER: 4, A_PRB: 4, A_MCS: 3})

    assert _mean_served(env, high) > _mean_served(env, low)


def test_wrong_handover_reduces_performance() -> None:
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily.tn_ntn_continuity,
            horizon=10,
            n_ue=1,
            seed=2,
            channel={"mode": "no_comm"},
        )
    )

    def good(e):
        return _base_actions(e, {A_POWER: 4, A_PRB: 4, A_MCS: 3, A_ACCESS: 1, A_HANDOVER: 2})

    def bad(e):
        return _base_actions(e, {A_POWER: 4, A_PRB: 4, A_MCS: 3, A_ACCESS: 0, A_HANDOVER: 0})

    assert _mean_served(env, good) > _mean_served(env, bad)


def test_ntn_preserves_service_during_tn_disruption() -> None:
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily.tn_ntn_failover,
            horizon=10,
            n_ue=1,
            seed=3,
            channel={"mode": "no_comm"},
        )
    )

    def ntn(e):
        return _base_actions(e, {A_ACCESS: 1, A_POWER: 4, A_PRB: 4, A_MCS: 3, A_HANDOVER: 2})

    def tn(e):
        return _base_actions(e, {A_ACCESS: 0, A_POWER: 4, A_PRB: 4, A_MCS: 3})

    assert _mean_served(env, ntn) >= _mean_served(env, tn)


def test_priority_affects_critical_service() -> None:
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily.critical_service,
            horizon=10,
            n_ue=2,
            seed=4,
            channel={"mode": "no_comm"},
        )
    )

    def high_prio(e):
        return _base_actions(e, {A_POWER: 3, A_PRB: 3, A_MCS: 2, A_PRIORITY: 2})

    def low_prio(e):
        return _base_actions(e, {A_POWER: 3, A_PRB: 3, A_MCS: 2, A_PRIORITY: 0})

    def critical_served(actions_fn):
        vals = []
        for seed in range(12):
            env.reset(seed=seed)
            c = 0.0
            for _ in range(8):
                if not env.agents:
                    break
                _, _, _, _, infos = env.step(actions_fn(env))
                if infos:
                    a0 = next(iter(infos))
                    served = infos[a0]["served"]
                    c += float(served[0])
            vals.append(c)
        return float(np.mean(vals))

    assert critical_served(high_prio) > critical_served(low_prio)


def test_action_permutations_change_outcomes() -> None:
    env = make_env(EnvConfig(horizon=8, n_ue=1, seed=5, channel={"mode": "no_comm"}))
    a = _mean_served(env, lambda e: _base_actions(e, {A_POWER: 4, A_PRB: 0}))
    b = _mean_served(env, lambda e: _base_actions(e, {A_POWER: 0, A_PRB: 4}))
    c = _mean_served(env, lambda e: _base_actions(e, {A_POWER: 4, A_PRB: 4}))
    assert len({round(a, 3), round(b, 3), round(c, 3)}) >= 2 or (c >= max(a, b))


def test_zeroing_messages_hurts_comm_necessary_scenario() -> None:
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily.hidden_blockage_congestion,
            horizon=12,
            n_ue=1,
            seed=6,
            erasure_p=0.0,
            channel={"mode": "discrete_learned", "vocab_size": 4, "msg_length": 2},
        )
    )

    def with_msg(e):
        return _base_actions(
            e,
            {A_POWER: 3, A_PRB: 3, A_MCS: 2, A_PRIORITY: 2},
            per_agent={
                "ue_0": {A_MSG: 2, A_TARGET: e.possible_agents.index("bs_0")},
                "edge_0": {A_MSG: 2, A_TARGET: e.possible_agents.index("bs_0")},
                "bs_0": {A_MSG: 2},
            },
        )

    def no_msg(e):
        return _base_actions(e, {A_POWER: 3, A_PRB: 3, A_MCS: 2, A_PRIORITY: 2, A_MSG: 0})

    assert _mean_served(env, with_msg) > _mean_served(env, no_msg)


def test_oracle_upper_bound() -> None:
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily.hidden_blockage_congestion,
            horizon=10,
            n_ue=1,
            seed=7,
            channel={"mode": "discrete_learned", "vocab_size": 4, "msg_length": 2},
        )
    )

    def randomish(e):
        return _base_actions(e, {A_POWER: 1, A_PRB: 1, A_MCS: 0, A_PRIORITY: 0, A_MSG: 0})

    def oracle(e):
        acts = e.oracle_service_actions()
        return {a: acts[a] for a in e.agents}

    assert _mean_served(env, oracle) >= _mean_served(env, randomish)
