"""Integration and scenario schema tests."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import yaml

from emergent_intent.env import EnvConfig, make_env
from emergent_intent.intent import ServiceIntent, intent_json_schema
from emergent_intent.interpretability import analyze_messages


ROOT = Path(__file__).resolve().parents[2]


def test_scenario_yaml_loads() -> None:
    for path in (ROOT / "configs/scenarios").glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        cfg = EnvConfig.from_dict(data)
        env = make_env(cfg)
        obs, _ = env.reset(seed=0)
        assert len(obs) >= 3


def test_service_intent_json_schema() -> None:
    _ = intent_json_schema()
    instance = {
        "service_id": "edu-1",
        "service_class": "education",
        "priority": 2,
        "max_latency_ms": 50,
        "min_reliability": 0.95,
        "fairness_floor": 0.7,
        "energy_budget": 1.0,
        "description": "test",
        "constraints": [],
    }
    intent = ServiceIntent.model_validate(instance)
    assert intent.service_id == "edu-1"
    file_schema = json.loads((ROOT / "schemas/service_intent.schema.json").read_text())
    jsonschema.validate(instance, file_schema)


def test_interpretability_bundle() -> None:
    msgs = np.random.randint(0, 8, size=(50, 2))
    cond = np.random.randn(50)
    meanings = np.random.randn(50, 2)
    report = analyze_messages(msgs, conditions=cond, vocab_size=8, meanings=meanings)
    assert "entropy_bits" in report
    assert report["claim_level"] == "descriptive_statistics_only"
    assert "emergent language" in report["warning"].lower()
