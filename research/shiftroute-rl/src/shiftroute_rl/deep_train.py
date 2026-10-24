"""Supervised fine-tuning -> clipped PPO fine-tuning with explicit train/eval split."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .deep_env import sample_context, tool_features, dynamics, HORIZON, intended_action
from .deep_models import ActorCritic, RiskMonitor

@dataclass
class TrainConfig:
    seed: int = 7
    monitor_samples: int = 12000
    monitor_epochs: int = 15
    sft_steps: int = 220
    ppo_updates: int = 190
    batch_episodes: int = 128
    learning_rate: float = .0015
    clip_eps: float = .20
    entropy_coeff: float = .015
    gamma: float = .96
    cert_rate: float = .62


def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(3)


def fit_monitor(cfg: TrainConfig) -> tuple[RiskMonitor, dict]:
    rng = np.random.default_rng(cfg.seed + 99)
    n = cfg.monitor_samples
    obs = sample_context(rng, n, 1)[:, 0, :]
    # Training deliberately over-represents easy obvious hacks.
    targets = intended_action(obs)
    a = targets.copy()
    unsafe = rng.random(n) < .42
    categories = rng.choice([0,1,2],size=n,p=[.45,.38,.17])
    a[unsafe & (categories==0)] = (targets[unsafe & (categories==0)]+1)%4
    a[unsafe & (categories==1)] = 4
    a[unsafe & (categories==2)] = 5
    x = torch.tensor(tool_features(obs, a, rng=rng))
    y = torch.tensor((a != targets).astype(np.float32))
    model = RiskMonitor()
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    criterion = nn.BCEWithLogitsLoss()
    # Offline split by generation index. Calibration and test data use distinct RNGs.
    perm = rng.permutation(n)
    cut = int(.8 * n)
    train_idx = torch.tensor(perm[:cut]); val_idx = torch.tensor(perm[cut:])
    history = []
    for epoch in range(cfg.monitor_epochs):
        model.train()
        for batch in train_idx.split(512):
            pred = model(x[batch]); loss = criterion(pred, y[batch])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(criterion(model(x[val_idx]), y[val_idx]))
            probs = model(x[val_idx]).sigmoid()
            acc = float(((probs >= .5) == y[val_idx].bool()).float().mean())
        history.append({"epoch":epoch + 1, "val_loss":vloss, "val_acc":acc})
    model.eval()
    return model, {"train_count":cut, "val_count":n - cut, "history":history}


def sft_actor(cfg: TrainConfig) -> ActorCritic:
    """Train a valid-action imitation policy before RL updates (actual SFT)."""
    rng = np.random.default_rng(cfg.seed + 155)
    actor = ActorCritic()
    opt = torch.optim.AdamW(actor.parameters(), lr=.002)
    for step in range(cfg.sft_steps):
        x = torch.from_numpy(sample_context(rng, 128, 1)[:, 0, :])
        logits, _ = actor(x)
        target = ((x[:,:4].argmax(dim=-1) + x[:,4:7].argmax(dim=-1)) % 4)
        loss = nn.functional.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
    return actor


def calculate_gae(rewards: torch.Tensor, values: torch.Tensor,
                  gamma: float, lam: float = .95) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation over the full multi-step trajectories."""
    adv = torch.zeros_like(rewards)
    last = torch.zeros(rewards.shape[0])
    for t in reversed(range(rewards.shape[1])):
        next_value = values[:, t + 1] if t + 1 < values.shape[1] else torch.zeros_like(last)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    return adv, adv + values


def ppo_train(cfg: TrainConfig, mode: str, monitor: RiskMonitor,
              sft_weights: dict, output_dir: Path) -> tuple[ActorCritic, list[dict]]:
    rng = np.random.default_rng(cfg.seed + 500)
    actor = ActorCritic()
    actor.load_state_dict(sft_weights)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.learning_rate)
    history = []
    for update in range(cfg.ppo_updates):
        obs = sample_context(rng, cfg.batch_episodes)
        # Sequential rollouts: prior accepted actions change later policy input.
        accepted = np.zeros(cfg.batch_episodes, dtype=np.float32)
        actions_list, logps, vals_list = [], [], []
        for t in range(HORIZON):
            obs[:,t,11] = accepted / HORIZON
            obs[:,t,12] = t / HORIZON
            with torch.no_grad():
                logits_t, values_t = actor(torch.from_numpy(obs[:,t,:]))
                dist_t = Categorical(logits=logits_t)
                action_t = dist_t.sample()
                actions_list.append(action_t)
                logps.append(dist_t.log_prob(action_t))
                vals_list.append(values_t)
            act = action_t.numpy()
            # Proxy scorer treats all purported completions as progress.
            # Independently verified modes only credit legitimate actions.
            expert=intended_action(obs[:,t,:])
            accepted += np.ones_like(accepted) if mode == 'proxy' else (act == expert).astype(np.float32)
        act_tensor = torch.stack(actions_list, 1)
        np_actions = act_tensor.numpy()
        with torch.no_grad():
            f = tool_features(obs, np_actions, rng=rng)
            mp = monitor(torch.tensor(f.reshape(-1, 8))).sigmoid().numpy().reshape(np_actions.shape)
            rewards_np, diag = dynamics(obs, np_actions, mp, mode, rng=rng,
                                        cert_rate=cfg.cert_rate)
            # True, multi-step success bonus is ONLY available to verified rewards.
            if mode != 'proxy':
                rewards_np[:,-1] += 1.20 * np.all(np_actions == intended_action(obs),axis=1).astype(np.float32)
            rewards = torch.tensor(rewards_np)
            vals = torch.stack(vals_list,1)
            advantage, returns = calculate_gae(rewards, vals, cfg.gamma)
            advantage = (advantage - advantage.mean())/(advantage.std()+1e-8)
            advantage = advantage.flatten()
            returns = returns.flatten()
            flat = torch.from_numpy(obs.reshape(-1, obs.shape[-1]))
            actions = act_tensor.flatten()
            old_logp = torch.stack(logps,1).flatten()
        # Multiple updates on the SAME rollouts: actual clipped PPO optimization.
        for _ in range(3):
            newlogits, newvals = actor(flat)
            dist = Categorical(logits=newlogits)
            ratio = torch.exp(dist.log_prob(actions) - old_logp)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1-cfg.clip_eps, 1+cfg.clip_eps) * advantage
            loss = -torch.minimum(unclipped, clipped).mean() + .3 * (newvals-returns).square().mean() - cfg.entropy_coeff*dist.entropy().mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), .8)
            optimizer.step()
        if (update+1) % 10 == 0 or update == 0:
            history.append({"update":update+1,
                "train_mean_proxy_reward":float(rewards_np.mean()),
                "train_shortcut_rate":float((np_actions!=intended_action(obs)).mean()),
                "train_subtle_rate":float((np_actions==5).mean()),
                "train_unsafe_execution_rate":float(diag["executed_unsafe"].mean()),
                "train_verification_rate":float(diag["verified"].mean())})
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict":actor.state_dict(),"config":asdict(cfg),"mode":mode},output_dir/f"policy_{mode}_seed{cfg.seed}.pt")
    return actor, history


def run_train(cfg: TrainConfig, output_dir: Path):
    seed_all(cfg.seed)
    output_dir.mkdir(parents=True,exist_ok=True)
    monitor, hist = fit_monitor(cfg)
    torch.save({"state_dict":monitor.state_dict(),"config":asdict(cfg)},output_dir/f"monitor_seed{cfg.seed}.pt")
    base = sft_actor(cfg)
    torch.save({"state_dict":base.state_dict(),"config":asdict(cfg)},output_dir/f"sft_seed{cfg.seed}.pt")
    histories = {"monitor":hist, "train":{}}
    for mode in ("proxy", "verified", "capability"):
        actor, h=ppo_train(cfg, mode, monitor,base.state_dict(),output_dir)
        histories["train"][mode]=h
    (output_dir/f"training_seed{cfg.seed}.json").write_text(json.dumps(histories,indent=2))
    return histories
