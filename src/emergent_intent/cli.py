"""CLI entrypoints for smoke / train / eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emergent_intent.algorithms import PPOConfig, make_trainer
from emergent_intent.env import EnvConfig, make_env
from emergent_intent.utils import detect_device, dump_json, load_yaml, new_manifest, set_global_seed


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_smoke(config_path: Path | None = None, seed: int = 0, out_dir: Path | None = None) -> dict:
    root = _repo_root()
    config_path = config_path or root / "configs" / "smoke" / "cpu_smoke.yaml"
    out_dir = out_dir or root / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(config_path)
    set_global_seed(seed)
    env_cfg = EnvConfig.from_dict(cfg.get("env", cfg))
    env_cfg.seed = seed
    env = make_env(env_cfg)
    algo = cfg.get("algorithm", "ippo")
    steps = int(cfg.get("total_steps", 256))
    ppo = PPOConfig(
        rollout_steps=int(cfg.get("rollout_steps", 64)),
        epochs=int(cfg.get("epochs", 2)),
        hidden=int(cfg.get("hidden", 32)),
    )
    trainer_kwargs: dict = {"seed": seed, "prefer_cuda": bool(cfg.get("prefer_cuda", False))}
    if algo.lower() in ("ippo", "mappo", "dial", "tarmac", "dial_tarmac"):
        trainer_kwargs["config"] = ppo
    if algo.lower() in ("dial", "tarmac", "dial_tarmac"):
        ch = cfg.get("env", {}).get("channel", {})
        trainer_kwargs["vocab_size"] = int(ch.get("vocab_size", env_cfg.vocab_size))
        trainer_kwargs["msg_length"] = int(ch.get("msg_length", env_cfg.msg_len))

    manifest = new_manifest(
        experiment_id=f"smoke_{config_path.stem}_{algo}_seed{seed}",
        config_path=str(config_path),
        seed=seed,
        evidence_class=cfg.get("evidence_class", "SYNTHETIC_SIM"),
    )
    trainer = make_trainer(algo, env, **trainer_kwargs)
    metrics = trainer.train(total_steps=steps)
    env2 = make_env(env_cfg)
    obs, _ = env2.reset(seed=seed + 999)
    eval_r = 0.0
    for _ in range(env_cfg.horizon):
        if hasattr(trainer, "select_actions"):
            out = trainer.select_actions(obs)
            actions = out[0] if isinstance(out, tuple) else out
        else:
            actions = {a: env2.action_space(a).sample() for a in obs}
        obs, rewards, terms, truncs, _ = env2.step(actions)
        if rewards:
            eval_r += float(sum(rewards.values()) / len(rewards))
        if not env2.agents:
            break
    metrics["eval_return"] = eval_r
    metrics["episode_summary"] = env2.episode_summary()
    manifest.finalize("SUCCESS", metrics)
    out_path = out_dir / f"{manifest.experiment_id}.json"
    manifest.save(out_path)
    ckpt = out_dir / f"{manifest.experiment_id}.pt"
    if hasattr(trainer, "save"):
        trainer.save(ckpt)
    return manifest.to_dict()


def smoke_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="CPU/GPU smoke runner")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args(argv)
    info = detect_device(prefer_cuda=True)
    if args.require_cuda and not info.cuda_available:
        payload = {
            "status": "BLOCKED_HARDWARE",
            "evidence_class": "BLOCKED",
            "reason": "CUDA required but torch.cuda.is_available() is False",
            "device": info.as_dict(),
        }
        out = args.out or _repo_root() / "results" / "smoke" / "gate4_gpu_blocked.json"
        dump_json(out, payload)
        print(json.dumps(payload, indent=2))
        raise SystemExit(2)
    result = run_smoke(args.config, seed=args.seed, out_dir=args.out)
    print(json.dumps(result, indent=2))


def train_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    result = run_smoke(args.config, seed=args.seed, out_dir=args.out)
    print(json.dumps(result, indent=2))


def eval_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    cfg = load_yaml(args.config)
    env = make_env(EnvConfig.from_dict(cfg.get("env", cfg)))
    obs, _ = env.reset(seed=args.seed)
    total = 0.0
    for _ in range(env.config.horizon):
        actions = {a: env.action_space(a).sample() for a in obs}
        obs, rewards, _, _, _ = env.step(actions)
        if rewards:
            total += float(sum(rewards.values()) / len(rewards))
        if not env.agents:
            break
    print(json.dumps({"eval_return": total, "summary": env.episode_summary()}, indent=2))
