"""Semantic intervention causality: messages help only via obs→actions."""

from __future__ import annotations

import numpy as np
import pytest

from emergent_intent.comm.interventions import (
    InterventionKind,
    apply_post_step_inbox_intervention,
    wrap_fixed_protocol_encoder,
)
from emergent_intent.comm.semantic_protocol import (
    DEFAULT_MAPPING,
    SYMBOL,
    SemanticProtocolController,
    encode_fixed_protocol_message,
)
from emergent_intent.env import EnvConfig, ScenarioFamily, make_env


def _mean_return(
    scenario: ScenarioFamily,
    kind: InterventionKind,
    *,
    seeds: tuple[int, ...] = tuple(range(6)),
    horizon: int = 10,
    n_ue: int = 1,
) -> float:
    rng = np.random.default_rng(0)
    controller = SemanticProtocolController(max_symbol_age=5.0)
    totals = []
    for seed in seeds:
        env = make_env(
            EnvConfig(
                scenario=scenario,
                horizon=horizon,
                n_ue=n_ue,
                seed=seed,
                erasure_p=0.0,
                corruption_p=0.0,
                delay=2 if kind == InterventionKind.DELAY else 0,
                channel={
                    "mode": "fixed_protocol",
                    "vocab_size": 16,
                    "msg_length": 2,
                },
            )
        )
        assert env.message_presence_bonus_enabled is False
        # Source-side interventions via encoder wrap (single application — no double-flip).
        wrap_fixed_protocol_encoder(env, kind, rng)
        obs, _ = env.reset(seed=seed)
        ep = 0.0
        for t in range(horizon):
            if not env.agents:
                break
            # STALE: age inbox so controller filters symbols before acting
            if kind == InterventionKind.STALE and t > 0:
                apply_post_step_inbox_intervention(env, InterventionKind.STALE, rng)
            acts = controller.actions_from_inbox(env, getattr(env, "_last_inbox", {}))
            if kind == InterventionKind.SILENCE:
                from emergent_intent.env.wireless_env import A_MSG

                for a in acts:
                    acts[a][A_MSG] = 0
            obs, rewards, _, _, _ = env.step(acts)
            if rewards:
                ep += float(sum(rewards.values()) / max(len(rewards), 1))
        totals.append(ep)
    return float(np.mean(totals))


def test_protocol_mappings_cover_required_semantics() -> None:
    m = DEFAULT_MAPPING.as_dict()
    for key in ("blockage", "congestion", "tn_ntn", "priority", "handover"):
        assert key in m
    assert SYMBOL["blockage_high"] != SYMBOL["blockage_clear"]
    assert SYMBOL["tn_down"] != SYMBOL["tn_ok"]
    assert SYMBOL["priority_high"] != SYMBOL["priority_low"]
    assert SYMBOL["handover_needed"] != SYMBOL["handover_hold"]


def test_fixed_protocol_encodes_blockage_congestion_tn_ntn_priority_handover() -> None:
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily.hidden_blockage_congestion,
            horizon=4,
            n_ue=1,
            seed=0,
            channel={"mode": "fixed_protocol", "vocab_size": 16, "msg_length": 2},
        )
    )
    env.reset(seed=0)
    ue = encode_fixed_protocol_message("ue_0", env._state, msg_len=2)
    bs = encode_fixed_protocol_message("bs_0", env._state, msg_len=2)
    edge = encode_fixed_protocol_message("edge_0", env._state, msg_len=2)
    assert SYMBOL["blockage_high"] in set(ue.astype(int).tolist())
    assert SYMBOL["congestion_high"] in set(bs.astype(int).tolist()) or SYMBOL["congestion_low"] in set(
        bs.astype(int).tolist()
    )
    assert SYMBOL["priority_high"] in set(edge.astype(int).tolist()) or SYMBOL["handover_needed"] in set(
        edge.astype(int).tolist()
    ) or SYMBOL["handover_hold"] in set(edge.astype(int).tolist())


def test_no_message_presence_bonus_in_reward() -> None:
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily.hidden_blockage_congestion,
            horizon=3,
            n_ue=1,
            seed=1,
            channel={"mode": "fixed_protocol", "vocab_size": 16},
        )
    )
    env.reset(seed=1)
    controller = SemanticProtocolController()
    acts = controller.actions_from_inbox(env, {})
    _, _, _, _, infos = env.step(acts)
    a0 = next(iter(infos))
    assert infos[a0]["coordination"] == 0.0
    assert env.message_presence_bonus_enabled is False


@pytest.mark.parametrize(
    "kind",
    [
        InterventionKind.RANDOM,
        InterventionKind.CONSTANT,
        InterventionKind.PERMUTE,
        InterventionKind.SILENCE,
        InterventionKind.CORRUPT,
        InterventionKind.ADVERSARIAL,
        InterventionKind.STALE,
        InterventionKind.DELAY,
    ],
)
def test_interventions_hurt_or_not_help_vs_correct(kind: InterventionKind) -> None:
    correct = _mean_return(ScenarioFamily.hidden_blockage_congestion, InterventionKind.CORRECT)
    intervened = _mean_return(ScenarioFamily.hidden_blockage_congestion, kind)
    # Correct semantic protocol should beat destructive interventions (soft inequality:
    # delay/stale may be close; require correct >= intervened - epsilon OR strictly better
    # for clearly destructive kinds).
    if kind in (
        InterventionKind.SILENCE,
        InterventionKind.ADVERSARIAL,
        InterventionKind.RANDOM,
        InterventionKind.CONSTANT,
    ):
        assert correct > intervened
    else:
        assert correct >= intervened - 1e-6


def test_correct_beats_silence_on_tn_ntn() -> None:
    correct = _mean_return(ScenarioFamily.tn_ntn_continuity, InterventionKind.CORRECT, n_ue=1)
    silence = _mean_return(ScenarioFamily.tn_ntn_continuity, InterventionKind.SILENCE, n_ue=1)
    assert correct > silence


def test_messages_help_only_via_observation_action_path() -> None:
    """Without a controller reading inbox, extra messages alone do not raise reward."""
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily.hidden_blockage_congestion,
            horizon=8,
            n_ue=1,
            seed=3,
            erasure_p=0.0,
            channel={"mode": "fixed_protocol", "vocab_size": 16, "msg_length": 2},
        )
    )
    from emergent_intent.env.wireless_env import A_ADMISSION, A_MCS, A_POWER, A_PRB, A_PRIORITY

    def blind(e):
        acts = {a: np.zeros(len(e._nvec), dtype=np.int64) for a in e.agents}
        for a in acts:
            acts[a][A_ADMISSION] = 1
            acts[a][A_POWER] = 2
            acts[a][A_PRB] = 2
            acts[a][A_MCS] = 1
            acts[a][A_PRIORITY] = 1
        return acts

    def run(actions_fn):
        totals = []
        for seed in range(5):
            env.reset(seed=seed)
            ep = 0.0
            for _ in range(8):
                if not env.agents:
                    break
                _, rewards, _, _, infos = env.step(actions_fn(env))
                assert all(infos[a]["coordination"] == 0.0 for a in infos)
                ep += float(sum(rewards.values()) / max(len(rewards), 1))
            totals.append(ep)
        return float(np.mean(totals))

    blind_ret = run(blind)
    controller = SemanticProtocolController()
    # Warm controller path
    ctrl_rets = []
    for seed in range(5):
        env.reset(seed=seed)
        ep = 0.0
        for _ in range(8):
            if not env.agents:
                break
            acts = controller.actions_from_inbox(env, env._last_inbox)
            _, rewards, _, _, infos = env.step(acts)
            assert all(infos[a]["coordination"] == 0.0 for a in infos)
            ep += float(sum(rewards.values()) / max(len(rewards), 1))
        ctrl_rets.append(ep)
    assert float(np.mean(ctrl_rets)) > blind_ret
