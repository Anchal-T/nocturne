import random
import threading
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Q-network.

    A CNN encoder processes the flattened occupancy-grid portion of the
    observation; its output is concatenated with the remaining ego/path/TTZ
    features before passing through a small MLP head.

    ``hidden_layers`` is a 4-element list to preserve config compatibility:
      [0] unused
      [1] encoder output width  (Linear after the conv stack)
      [2] unused
      [3] head hidden width
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        grid_size: int,
        hidden_layers: Optional[List[int]] = None,
        grid_rows: int = 25,
        grid_cols: int = 14,
    ):
        super().__init__()
        hidden_layers = hidden_layers or [128, 128, 128, 32]
        assert len(hidden_layers) == 4

        self.hidden_layers = hidden_layers
        self.grid_size = grid_size
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        extra_dim = obs_dim - grid_size

        # Build the conv backbone first so we can probe its output size with a
        # dummy tensor rather than hard-coding 32 * 6 * 3 for a fixed 25×14 grid.
        conv_backbone = nn.Sequential(
            nn.Unflatten(1, (1, grid_rows, grid_cols)),
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = conv_backbone(torch.zeros(1, grid_size)).shape[1]

        self.grid_encoder = nn.Sequential(
            *conv_backbone,
            nn.Linear(flat_dim, hidden_layers[1]),
            nn.ReLU(),
        )

        # Head input is encoder output (hidden_layers[1]) + non-grid features.
        # hidden_layers[2] is intentionally unused; [1] is the actual encoder width.
        self.head = nn.Sequential(
            nn.Linear(hidden_layers[1] + extra_dim, hidden_layers[3]),
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
        grid_rows: int = 25,
        grid_cols: int = 14,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        def _make_net():
            return QNetwork(obs_dim, n_actions, grid_size, hidden_layers, grid_rows, grid_cols).to(self.device)

        self.online_net = _make_net()
        self.target_net = _make_net()
        self.inference_net = _make_net()
        self.inference_lock = threading.Lock()
        self.sync_target()
        self.sync_inference_net()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.train_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with self.inference_lock:
                q_values = self.inference_net(state_t)
            return q_values.argmax(dim=1).item()

    def select_action_batch(self, states: np.ndarray) -> np.ndarray:
        batch_size = states.shape[0]
        actions = np.zeros(batch_size, dtype=int)
        
        random_mask = np.random.rand(batch_size) < self.epsilon
        
        if random_mask.any():
            actions[random_mask] = np.random.randint(0, self.n_actions, size=random_mask.sum())
            
        greedy_mask = ~random_mask
        if greedy_mask.any():
            with torch.no_grad():
                states_t = torch.FloatTensor(states[greedy_mask]).to(self.device)
                with self.inference_lock:
                    q_values = self.inference_net(states_t)
                actions[greedy_mask] = q_values.argmax(dim=1).cpu().numpy()
                
        return actions

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def store_transition_batch(self, states, actions, rewards, next_states, dones):
        self.replay_buffer.buffer.extend(zip(states, actions, rewards, next_states, dones))

    def train_step(self, env_steps: int = 0) -> Optional[float]:
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
            best_actions = self.online_net(next_states_t).argmax(dim=1)
            next_q = (
                self.target_net(next_states_t)
                .gather(1, best_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        self._decay_epsilon(env_steps)

        if self.train_steps % self.target_update_freq == 0:
            self.sync_target()

        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def sync_inference_net(self):
        with self.inference_lock:
            self.inference_net.load_state_dict(self.online_net.state_dict())

    def _decay_epsilon(self, env_steps: int):
        if env_steps <= 0:
            return
        fraction = min(1.0, env_steps / max(1, self.epsilon_decay_steps))
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
                "grid_rows": self.grid_rows,
                "grid_cols": self.grid_cols,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # If the checkpoint stores architecture, rebuild networks to match before loading weights.
        if "hidden_layers" in checkpoint:
            ckpt_hl = checkpoint["hidden_layers"]
            ckpt_rows = checkpoint.get("grid_rows", self.grid_rows)
            ckpt_cols = checkpoint.get("grid_cols", self.grid_cols)
            needs_rebuild = (
                ckpt_hl != self.online_net.hidden_layers
                or ckpt_rows != self.grid_rows
                or ckpt_cols != self.grid_cols
            )
            if needs_rebuild:
                self.online_net = QNetwork(
                    self.obs_dim, self.n_actions, self.online_net.grid_size, ckpt_hl,
                    ckpt_rows, ckpt_cols,
                ).to(self.device)
                self.target_net = QNetwork(
                    self.obs_dim, self.n_actions, self.target_net.grid_size, ckpt_hl,
                    ckpt_rows, ckpt_cols,
                ).to(self.device)
                self.inference_net = QNetwork(
                    self.obs_dim, self.n_actions, self.inference_net.grid_size, ckpt_hl,
                    ckpt_rows, ckpt_cols,
                ).to(self.device)
                self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.optimizer.param_groups[0]["lr"])
            self.grid_rows = ckpt_rows
            self.grid_cols = ckpt_cols
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.sync_inference_net()
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint["train_steps"]
        self.epsilon = checkpoint["epsilon"]
