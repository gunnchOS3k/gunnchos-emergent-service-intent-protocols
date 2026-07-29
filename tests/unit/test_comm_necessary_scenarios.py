"""§6.3 communication-necessary scenario documentation & asymmetry tests."""

from __future__ import annotations

import numpy as np

from emergent_intent.env import EnvConfig, ScenarioFamily, make_env
from emergent_intent.env.wireless_env import LOCAL_OBS_DIM


def test_scenario_a_hidden_blockage_asymmetry() -> None:
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily.hidden_blockage_congestion,
            horizon=4,
            n_ue=2,
            seed=0,
            channel={"mode": "no_comm"},
        )
    )
    obs, infos = env.reset(seed=0)
    # UE sees high local blockage; BS does not get full UE blockage (obs feature forced 0)
    assert obs["ue_0"][4] > 0.8  # blockage feature
    assert abs(obs["bs_0"][7]) < 1e-6  # blockage hidden from BS in scenario A
    # Edge sees priority / critical sum
    assert obs["edge_0"][6] >= 1.0
    note = infos["ue_0"]["comm_necessity"]
    assert "blockage" in note.lower() or "messaging" in note.lower()


def test_scenario_b_tn_ntn_continuity_asymmetry() -> None:
    env = make_env(
        EnvConfig(
            scenario=ScenarioFamily.tn_ntn_continuity,
            horizon=4,
            n_ue=1,
            seed=1,
            channel={"mode": "no_comm"},
        )
    )
    obs, infos = env.reset(seed=1)
    assert "ntn_relay" in env.possible_agents
    assert env._state["tn_available"] < 0.3
    # NTN sees cost; UE does not get full ntn_cost in local features (index 9 near 0)
    assert obs["ntn_relay"][2] > 0.0  # ntn_cost
    assert "failover" in infos["ue_0"]["comm_necessity"].lower() or "communication" in infos[
        "ue_0"
    ]["comm_necessity"].lower()


def test_no_comm_information_disadvantage_documented() -> None:
    """No-comm baseline cannot obtain complementary hidden state via inbox."""
    for sc in (ScenarioFamily.hidden_blockage_congestion, ScenarioFamily.tn_ntn_continuity):
        env = make_env(EnvConfig(scenario=sc, horizon=3, n_ue=1, seed=2, channel={"mode": "no_comm"}))
        obs, _ = env.reset(seed=2)
        for a in obs:
            assert np.allclose(obs[a][LOCAL_OBS_DIM:], 0.0)
