"""Dueling Double DQN agent with a uniform replay buffer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    """Q network with separate state value and action advantage streams."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        value = self.value(h)
        adv = self.advantage(h)
        return value + adv - adv.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    """Fixed size ring buffer over numpy arrays."""

    def __init__(self, capacity: int, state_dim: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add(self, state, action, reward, next_state, done) -> None:
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
        )


class D3QNAgent:
    """One independent learner: dueling network, double DQN targets."""

    def __init__(self, state_dim: int, n_actions: int, cfg: dict,
                 rng: np.random.Generator, device: str = "cpu"):
        self.cfg = cfg
        self.n_actions = n_actions
        self.rng = rng
        self.device = device
        self.gamma = float(cfg["gamma"])

        hidden = int(cfg["hidden_dim"])
        self.online = DuelingDQN(state_dim, n_actions, hidden).to(device)
        self.target = DuelingDQN(state_dim, n_actions, hidden).to(device)
        self.sync_target()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(cfg["lr"]))
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(cfg["buffer_size"], state_dim, rng)

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.online(torch.from_numpy(np.asarray(state, dtype=np.float32))
                        .unsqueeze(0).to(self.device))
        return int(q.argmax(dim=-1).item())

    def store(self, state, action, reward, next_state, done) -> None:
        self.buffer.add(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        batch_size = int(self.cfg["batch_size"])
        if self.buffer.size < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = states.to(self.device)
        next_states = next_states.to(self.device)

        q = self.online(states).gather(1, actions.to(self.device).unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.cfg.get("double", True):
                best = self.online(next_states).argmax(dim=1, keepdim=True)
                q_next = self.target(next_states).gather(1, best).squeeze(1)
            else:
                q_next = self.target(next_states).max(dim=1).values
            targets = rewards.to(self.device) + self.gamma * (1 - dones.to(self.device)) * q_next

        loss = self.loss_fn(q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
