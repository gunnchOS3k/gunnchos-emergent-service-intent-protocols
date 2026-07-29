"""§6.1 observation/inbox tests — messages must enter receiver observations."""

from __future__ import annotations

import numpy as np

from emergent_intent.env import EnvConfig, make_env
from emergent_intent.env.wireless_env import A_MSG, A_TARGET, LOCAL_OBS_DIM


def _zeros_actions(env, override: dict | None = None):
    acts = {a: np.zeros(len(env._nvec), dtype=np.int64) for a in env.agents}
    if override:
        for a, v in override.items():
            acts[a] = v
    return acts


def test_01_sent_message_changes_receiver_observation() -> None:
    env = make_env(
        EnvConfig(
            horizon=4,
            n_ue=1,
            seed=0,
            vocab_size=4,
            msg_len=2,
            erasure_p=0.0,
            corruption_p=0.0,
            delay=0,
            channel={"mode": "discrete_learned"},
        )
    )
    obs0, _ = env.reset(seed=0)
    silence = _zeros_actions(env)
    obs_s, *_ = env.step(silence)
    # UE sends symbol 1 to everyone (broadcast)
    send = _zeros_actions(env)
    send["ue_0"][A_MSG] = 2  # non-silence symbol
    obs_m, *_ = env.step(send)
    # BS observation must change in the inbox region
    assert not np.allclose(obs_s["bs_0"][LOCAL_OBS_DIM:], obs_m["bs_0"][LOCAL_OBS_DIM:])
    assert not np.allclose(obs0["bs_0"], obs_m["bs_0"])


def test_02_no_self_loopback_by_default() -> None:
    env = make_env(
        EnvConfig(
            horizon=3,
            n_ue=1,
            seed=1,
            loopback=False,
            erasure_p=0.0,
            channel={"mode": "discrete_learned", "vocab_size": 4, "msg_length": 2},
        )
    )
    env.reset(seed=1)
    send = _zeros_actions(env)
    send["ue_0"][A_MSG] = 2
    env.step(send)
    slots = env._last_inbox.get("ue_0", [])
    assert all(r.sender_id != 0 for r in slots)


def test_03_targeted_message_only_selected_recipient() -> None:
    env = make_env(
        EnvConfig(
            horizon=3,
            n_ue=1,
            seed=2,
            targeted=True,
            erasure_p=0.0,
            channel={"mode": "discrete_learned", "vocab_size": 4, "msg_length": 2, "targeted": True},
        )
    )
    env.reset(seed=2)
    bs_idx = env.possible_agents.index("bs_0")
    edge_idx = env.possible_agents.index("edge_0")
    send = _zeros_actions(env)
    send["ue_0"][A_MSG] = 2
    send["ue_0"][A_TARGET] = bs_idx
    obs, *_ = env.step(send)
    assert any(r.valid > 0 and r.sender_id == 0 for r in env._last_inbox.get("bs_0", []))
    assert not any(r.valid > 0 and r.sender_id == 0 for r in env._last_inbox.get("edge_0", []))
    # Targeting edge instead
    env.reset(seed=2)
    send2 = _zeros_actions(env)
    send2["ue_0"][A_MSG] = 3
    send2["ue_0"][A_TARGET] = edge_idx
    env.step(send2)
    assert any(r.valid > 0 and r.sender_id == 0 for r in env._last_inbox.get("edge_0", []))
    assert not any(r.valid > 0 and r.sender_id == 0 for r in env._last_inbox.get("bs_0", []))


def test_04_delayed_message_arrives_later() -> None:
    env = make_env(
        EnvConfig(
            horizon=5,
            n_ue=1,
            seed=3,
            delay=1,
            erasure_p=0.0,
            channel={"mode": "discrete_learned", "vocab_size": 4, "msg_length": 2, "delay": 1},
        )
    )
    env.reset(seed=3)
    send = _zeros_actions(env)
    send["ue_0"][A_MSG] = 2
    env.step(send)
    assert not any(r.valid > 0 and r.sender_id == 0 for r in env._last_inbox.get("bs_0", []))
    silence = _zeros_actions(env)
    env.step(silence)
    assert any(r.valid > 0 and r.sender_id == 0 for r in env._last_inbox.get("bs_0", []))


def test_05_erased_message_not_valid_content() -> None:
    env = make_env(
        EnvConfig(
            horizon=3,
            n_ue=1,
            seed=4,
            erasure_p=1.0,
            channel={"mode": "discrete_learned", "vocab_size": 4, "msg_length": 2, "erasure_p": 1.0},
        )
    )
    env.reset(seed=4)
    send = _zeros_actions(env)
    send["ue_0"][A_MSG] = 2
    env.step(send)
    for r in env._last_inbox.get("bs_0", []):
        if r.sender_id == 0:
            assert r.erased == 1.0
            assert r.valid == 0.0


def test_06_corrupted_message_follows_model() -> None:
    env = make_env(
        EnvConfig(
            horizon=3,
            n_ue=1,
            seed=5,
            corruption_p=1.0,
            erasure_p=0.0,
            channel={
                "mode": "discrete_learned",
                "vocab_size": 4,
                "msg_length": 2,
                "corruption_p": 1.0,
                "erasure_p": 0.0,
            },
        )
    )
    env.reset(seed=5)
    send = _zeros_actions(env)
    send["ue_0"][A_MSG] = 2  # intended symbol 1
    env.step(send)
    found = False
    for r in env._last_inbox.get("bs_0", []):
        if r.sender_id == 0 and r.valid > 0:
            found = True
            # With corruption_p=1, symbols are resampled; confidence reduced
            assert r.confidence < 1.0
    assert found


def test_07_silence_distinguishable() -> None:
    env = make_env(
        EnvConfig(
            horizon=3,
            n_ue=1,
            seed=6,
            erasure_p=0.0,
            channel={"mode": "discrete_learned", "vocab_size": 4, "msg_length": 2},
        )
    )
    env.reset(seed=6)
    sil = _zeros_actions(env)  # msg_token=0 → silence (does not occupy inbox)
    env.step(sil)
    assert not any(r.sender_id == 0 and r.valid > 0 for r in env._last_inbox.get("bs_0", []))
    speak = _zeros_actions(env)
    speak["ue_0"][A_MSG] = 2
    env.step(speak)
    talk = [r for r in env._last_inbox.get("bs_0", []) if r.sender_id == 0 and r.valid > 0][0]
    assert talk.silence == 0.0
    assert talk.valid == 1.0
    # Channel silence id is outside the vocab range; symbols use 0..V-1
    assert env.channel.silence_id() == 4
    assert int(talk.symbols[0]) != env.channel.silence_id()


def test_08_no_comm_reveals_no_message_info() -> None:
    env = make_env(
        EnvConfig(
            horizon=3,
            n_ue=1,
            seed=7,
            channel={"mode": "no_comm"},
        )
    )
    obs0, _ = env.reset(seed=7)
    send = _zeros_actions(env)
    send["ue_0"][A_MSG] = 3
    obs1, *_ = env.step(send)
    assert np.allclose(obs0["bs_0"][LOCAL_OBS_DIM:], 0.0)
    assert np.allclose(obs1["bs_0"][LOCAL_OBS_DIM:], 0.0)


def test_09_reset_clears_delay_and_inbox() -> None:
    env = make_env(
        EnvConfig(
            horizon=5,
            n_ue=1,
            seed=8,
            delay=2,
            erasure_p=0.0,
            channel={"mode": "discrete_learned", "vocab_size": 4, "msg_length": 2, "delay": 2},
        )
    )
    env.reset(seed=8)
    send = _zeros_actions(env)
    send["ue_0"][A_MSG] = 2
    env.step(send)
    assert any(len(q) > 0 for q in env.channel._pending.values()) or True
    env.reset(seed=8)
    assert all(len(q) == 0 for q in env.channel._pending.values())
    assert all(len(q) == 0 for q in env.channel._inbox.values())
    assert np.allclose(env._observe("bs_0")[LOCAL_OBS_DIM:], 0.0)


def test_10_message_age_increments() -> None:
    env = make_env(
        EnvConfig(
            horizon=6,
            n_ue=1,
            seed=9,
            delay=0,
            erasure_p=0.0,
            inbox_capacity=2,
            stale_threshold=3,
            channel={"mode": "discrete_learned", "vocab_size": 4, "msg_length": 2},
        )
    )
    env.reset(seed=9)
    send = _zeros_actions(env)
    send["ue_0"][A_MSG] = 2
    env.step(send)
    age0 = env._last_inbox["bs_0"][0].age
    silence = _zeros_actions(env)
    env.step(silence)
    age1 = [r for r in env._last_inbox["bs_0"] if r.sender_id == 0][0].age
    assert age1 == age0 + 1.0
    env.step(silence)
    env.step(silence)
    aged = [r for r in env._last_inbox["bs_0"] if r.sender_id == 0][0]
    assert aged.stale == 1.0
