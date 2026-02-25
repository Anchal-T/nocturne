import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Q-network matching the paper's architecture.

    First 3 hidden layers (128 each) process the occupancy grid portion.
    The processed features are then concatenated with ego/path/TTZ info
    before passing through the final 32-unit layer → Q-values.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        grid_size: int,
        hidden_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        hidden_layers = hidden_layers or [128, 128, 128, 32]
        assert len(hidden_layers) == 4

        self.hidden_layers = hidden_layers
        self.grid_size = grid_size
        extra_dim = obs_dim - grid_size

        self.grid_encoder = nn.Sequential(
            nn.Linear(grid_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_layers[2] + extra_dim, hidden_layers[3]),
            nn.ReLU(),
            nn.Linear(hidden_layers[3], n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid_input = x[:, : self.grid_size]
        extra_input = x[:, self.grid_size :]
        grid_features = self.grid_encoder(grid_input)
        combined = torch.cat([grid_features, extra_input], dim=1)
        return self.head(combined)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DDQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        grid_size: int = 350,
        hidden_layers: Optional[List[int]] = None,
        lr: float = 0.01,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 200000,
        replay_buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.online_net = QNetwork(obs_dim, n_actions, grid_size, hidden_layers).to(
            self.device
        )
        self.target_net = QNetwork(obs_dim, n_actions, grid_size, hidden_layers).to(
            self.device
        )
        self.sync_target()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.train_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        current_q = (
            self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            # DDQN: online net selects action, target net evaluates
            best_actions = self.online_net(next_states_t).argmax(dim=1)
            next_q = (
                self.target_net(next_states_t)
                .gather(1, best_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Clipping stabilizes training spikes when replay samples include large
        # reward deltas from collisions/goal bonuses.
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1
        # Epsilon decay is tied to optimizer updates (not env steps) so changing
        # train_freq automatically changes exploration decay speed.
        self._decay_epsilon()

        if self.train_steps % self.target_update_freq == 0:
            self.sync_target()

        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _decay_epsilon(self):
        fraction = min(1.0, self.train_steps / max(1, self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + fraction * (
            self.epsilon_end - self.epsilon_start
        )

    def save(self, path: str):
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
                "epsilon": self.epsilon,
                "hidden_layers": self.online_net.hidden_layers,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # If the checkpoint stores architecture, rebuild networks to match before loading weights.
        if "hidden_layers" in checkpoint:
            ckpt_hl = checkpoint["hidden_layers"]
            if ckpt_hl != self.online_net.hidden_layers:
                self.online_net = QNetwork(
                    self.obs_dim, self.n_actions, self.online_net.grid_size, ckpt_hl
                ).to(self.device)
                self.target_net = QNetwork(
                    self.obs_dim, self.n_actions, self.target_net.grid_size, ckpt_hl
                ).to(self.device)
                self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.optimizer.param_groups[0]["lr"])
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint["train_steps"]
        self.epsilon = checkpoint["epsilon"]
