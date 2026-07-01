"""Minimal independent PPO agent (clipped surrogate, GAE).

Custom and small on purpose: stable baselines does not fit a multi agent
custom loop cleanly, and the algorithm itself is short.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.policy = nn.Linear(hidden_dim, n_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.policy(h), self.value(h).squeeze(-1)


class PPOAgent:
    def __init__(self, state_dim: int, n_actions: int, cfg: dict,
                 rng: np.random.Generator, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.rng = rng
        self.net = ActorCritic(state_dim, n_actions, int(cfg["hidden_dim"])).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(cfg["lr"]))
        self.reset_rollout()

    def reset_rollout(self) -> None:
        self.states, self.actions, self.logps = [], [], []
        self.rewards, self.values, self.dones = [], [], []

    @torch.no_grad()
    def act(self, state: np.ndarray) -> int:
        x = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        logits, value = self.net(x.to(self.device))
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        action = int(self.rng.choice(len(probs), p=probs / probs.sum()))
        self._pending = (
            float(np.log(max(probs[action], 1e-12))), float(value.item())
        )
        return action

    def store(self, state, action, reward, done) -> None:
        logp, value = self._pending
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(action)
        self.logps.append(logp)
        self.rewards.append(float(reward))
        self.values.append(value)
        self.dones.append(float(done))

    def rollout_full(self) -> bool:
        return len(self.states) >= int(self.cfg["rollout_steps"])

    def update(self, last_state: np.ndarray) -> dict:
        cfg = self.cfg
        gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(last_state, dtype=np.float32)).unsqueeze(0)
            last_value = float(self.net(x.to(self.device))[1].item())

        rewards = np.asarray(self.rewards)
        values = np.asarray(self.values + [last_value])
        dones = np.asarray(self.dones)
        adv = np.zeros(len(rewards), dtype=np.float64)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states = torch.from_numpy(np.stack(self.states)).to(self.device)
        actions = torch.tensor(self.actions, dtype=torch.int64, device=self.device)
        old_logps = torch.tensor(self.logps, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        n = len(states)
        batch = int(cfg["minibatch_size"])
        losses = []
        for _ in range(int(cfg["update_epochs"])):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch):
                idx = perm[start : start + batch]
                logits, values_new = self.net(states[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions[idx])
                ratio = torch.exp(logp - old_logps[idx])
                clip = float(cfg["clip_range"])
                surrogate = torch.min(
                    ratio * adv_t[idx],
                    torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[idx],
                )
                policy_loss = -surrogate.mean()
                value_loss = ((values_new - ret_t[idx]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = (policy_loss
                        + float(cfg["value_coef"]) * value_loss
                        - float(cfg["entropy_coef"]) * entropy)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), float(cfg["max_grad_norm"]))
                self.optimizer.step()
                losses.append(float(loss.item()))

        self.reset_rollout()
        return {"loss": float(np.mean(losses))}

    def state_dict(self) -> dict:
        return {"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])
