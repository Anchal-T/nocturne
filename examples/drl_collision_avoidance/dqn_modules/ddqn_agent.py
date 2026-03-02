import random
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer

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
        dueling: bool = True,
        alpha = 0.6,
        beta_start = 0.4,
        beta_frames = 100000,
        per_epsilon = 1e-6,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 10.0,
        use_torch_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        inference_sync_interval: int = 4,
        profile_cuda: bool = False,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.dueling = bool(dueling)

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self._use_amp = self.device.type == "cuda"
        self._scaler = torch.amp.GradScaler(enabled=self._use_amp)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.max_grad_norm = float(max_grad_norm)
        self.inference_sync_interval = max(1, int(inference_sync_interval))
        self.profile_cuda = bool(profile_cuda)
        self._last_profile_log_step = 0
        self.compile_mode = str(compile_mode)
        self._compile_enabled = (
            bool(use_torch_compile)
            and self.device.type == "cuda"
            and hasattr(torch, "compile")
        )
        if bool(use_torch_compile) and self.device.type == "cuda" and not hasattr(torch, "compile"):
            print("[DDQNAgent] torch.compile unavailable in this PyTorch build; continuing without compile.")

        def _make_net() -> QNetwork:
            return QNetwork(
                obs_dim=obs_dim,
                n_actions=n_actions,
                grid_size=grid_size,
                hidden_layers=hidden_layers,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                dueling=self.dueling,
            ).to(self.device)

        self.online_net = _make_net()
        self.target_net = _make_net()
        self.inference_net = _make_net()
        self.online_net, self.target_net, self.inference_net = self._maybe_compile_all(
            self.online_net, self.target_net, self.inference_net
        )

        self.inference_lock = threading.Lock()
        self.sync_target()
        self.sync_inference_net()

        self.optimizer = self._build_optimizer(lr)

        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim,
            size=replay_buffer_size,
            batch_size=batch_size,
            n_step=1,
            gamma=gamma,
            alpha=alpha,
            beta_start=beta_start,
            beta_frames=beta_frames,
            epsilon=per_epsilon,
        )
        self.train_steps = 0
        self._grad_accum_counter = 0

        self._pin_buffers : Optional[dict] = None  # for optional pre-allocation of CUDA-pinned buffers to speed up batch sampling
        if self.device.type == "cuda":
            self._pin_buffers = {
                "obs" : torch.zeros(batch_size, obs_dim, pin_memory=True),
                "next_obs" : torch.zeros(batch_size, obs_dim, pin_memory=True),
                "acts" : torch.zeros(batch_size, dtype =torch.long, pin_memory=True),
                "rews" : torch.zeros(batch_size, pin_memory=True),
                "dones" : torch.zeros(batch_size, pin_memory=True),
                "weights" : torch.zeros(batch_size, pin_memory=True),
            }

    def _maybe_compile_all(
        self, online: nn.Module, target: nn.Module, inference: nn.Module
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        if not self._compile_enabled:
            return online, target, inference
        try:
            # Keep inference net eager: async actor path can hit FX tracing conflicts
            # with dynamo-optimized callables in some PyTorch builds.
            return (
                torch.compile(online, mode=self.compile_mode),
                torch.compile(target, mode=self.compile_mode),
                inference,
            )
        except Exception as exc:
            print(f"[DDQNAgent] torch.compile failed ({exc}); continuing in eager mode.")
            self._compile_enabled = False
            return online, target, inference

    def _build_optimizer(self, lr: float) -> optim.Optimizer:
        kwargs = {"lr": float(lr)}
        if self.device.type == "cuda":
            kwargs["fused"] = True
        try:
            return optim.Adam(self.online_net.parameters(), **kwargs)
        except TypeError:
            kwargs.pop("fused", None)
            return optim.Adam(self.online_net.parameters(), **kwargs)

    @staticmethod
    def _unwrap_module(module: nn.Module) -> nn.Module:
        if isinstance(module, nn.DataParallel):
            module = module.module
        orig_mod = getattr(module, "_orig_mod", None)
        if orig_mod is not None:
            module = orig_mod
        return module

    @staticmethod
    def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(state_dict)
        changed = True
        while changed and out:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if all(key.startswith(prefix) for key in out):
                    out = {key[len(prefix):]: value for key, value in out.items()}
                    changed = True
        return out

    @staticmethod
    def _add_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {f"{prefix}{key}": value for key, value in state_dict.items()}

    def _load_state_dict_flexible(self, module: nn.Module, state_dict: Dict[str, torch.Tensor], name: str) -> None:
        base = self._strip_known_prefixes(state_dict)
        candidates = [state_dict, base]
        candidates.append(self._add_prefix(base, "_orig_mod."))
        candidates.append(self._add_prefix(base, "module."))
        last_error = None
        for candidate in candidates:
            try:
                module.load_state_dict(candidate, strict=True)
                return
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(f"Failed to load {name} state dict across known key formats: {last_error}")

    def _maybe_log_cuda_profile(self) -> None:
        if not (self.profile_cuda and self.device.type == "cuda"):
            return
        if self.train_steps - self._last_profile_log_step < 200:
            return
        self._last_profile_log_step = self.train_steps
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        print(
            f"[CUDA] step={self.train_steps} mem_alloc={allocated:.1f}MiB "
            f"mem_reserved={reserved:.1f}MiB"
        )

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
        self.replay_buffer.store(
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )

    def store_transition_batch(self, states, actions, rewards, next_states, dones):
        self.replay_buffer.store_batch(states, actions, rewards, next_states, dones)

    def train_step(self, env_steps: int = 0) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample_batch()

        if self._pin_buffers is not None:
            # Reuse pinned buffers to avoid re-allocation
            pb = self._pin_buffers
            pb["obs"].copy_(torch.from_numpy(batch["obs"]))
            pb["next_obs"].copy_(torch.from_numpy(batch["next_obs"]))
            acts_np = batch["acts"]
            if acts_np.dtype != np.int64:
                acts_np = acts_np.astype(np.int64, copy=False)
            pb["acts"].copy_(torch.from_numpy(acts_np))
            pb["rews"].copy_(torch.from_numpy(batch["rews"]))
            pb["dones"].copy_(torch.from_numpy(batch["done"]))
            pb["weights"].copy_(torch.from_numpy(batch["weights"]))

            obs_t      = pb["obs"].to(self.device, non_blocking=True)
            next_obs_t = pb["next_obs"].to(self.device, non_blocking=True)
            acts_t     = pb["acts"].to(self.device, non_blocking=True)
            rews_t     = pb["rews"].to(self.device, non_blocking=True)
            dones_t    = pb["dones"].to(self.device, non_blocking=True)
            weights_t  = pb["weights"].to(self.device, non_blocking=True)
        else:
            obs_t      = torch.FloatTensor(batch["obs"]).to(self.device)
            next_obs_t = torch.FloatTensor(batch["next_obs"]).to(self.device)
            acts_t     = torch.LongTensor(batch["acts"]).to(self.device)
            rews_t     = torch.FloatTensor(batch["rews"]).to(self.device)
            dones_t    = torch.FloatTensor(batch["done"]).to(self.device)
            weights_t  = torch.FloatTensor(batch["weights"]).to(self.device)

        self.online_net.train()
        with torch.amp.autocast(device_type=self.device.type, enabled=self._use_amp):
            current_q = (
                self.online_net(obs_t)
                .gather(1, acts_t.unsqueeze(1))
                .squeeze(1)
            )
            with torch.no_grad():
                # DDQN: online selects action, target evaluates it
                best_acts = self.online_net(next_obs_t).argmax(dim=1)
                next_q    = (
                    self.target_net(next_obs_t)
                    .gather(1, best_acts.unsqueeze(1))
                    .squeeze(1)
                )
                target_q = rews_t + self.gamma * next_q * (1.0 - dones_t)

            td_errors = current_q - target_q                          # (B,)
            # IS-weighted Huber loss
            loss = (weights_t * F.huber_loss(current_q, target_q, reduction='none')).mean()

        scale = 1.0 / self.grad_accum_steps
        self._scaler.scale(loss * scale).backward()
        self._grad_accum_counter += 1

        if self._grad_accum_counter >= self.grad_accum_steps:
            self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.max_grad_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()
            self.optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
            self._grad_accum_counter = 0

        with torch.no_grad():
            new_priorities = td_errors.abs().cpu().numpy()
        self.replay_buffer.update_priorities(batch["indices"], new_priorities)
        self.train_steps += 1
        self.replay_buffer.update_beta(env_steps)
        self._decay_epsilon(env_steps)

        sync_inference = (self.train_steps % self.inference_sync_interval) == 0
        if self.train_steps % self.target_update_freq == 0:
            self.sync_target()
            sync_inference = True

        if sync_inference:
            self.sync_inference_net()

        self._maybe_log_cuda_profile()

        return float(loss.item())

    def sync_target(self):
        online_state = self._unwrap_module(self.online_net).state_dict()
        self._load_state_dict_flexible(self.target_net, online_state, "target_sync")

    def sync_inference_net(self):
        with self.inference_lock:
            online_state = self._unwrap_module(self.online_net).state_dict()
            self._load_state_dict_flexible(self.inference_net, online_state, "inference_sync")

    def _decay_epsilon(self, env_steps: int):
        if env_steps <= 0:
            return
        fraction = min(1.0, env_steps / max(1, self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + fraction * (
            self.epsilon_end - self.epsilon_start
        )

    def save(self, path: str):
        online_base = self._unwrap_module(self.online_net)
        target_base = self._unwrap_module(self.target_net)
        torch.save(
            {
                "online_net": online_base.state_dict(),
                "target_net": target_base.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
                "epsilon": self.epsilon,
                "hidden_layers": online_base.hidden_layers,
                "grid_size": online_base.grid_size,
                "grid_rows": self.grid_rows,
                "grid_cols": self.grid_cols,
                "dueling": online_base.dueling,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        online_base = self._unwrap_module(self.online_net)
        # If the checkpoint stores architecture, rebuild networks to match before loading weights.
        if "hidden_layers" in checkpoint:
            ckpt_hl = checkpoint["hidden_layers"]
            ckpt_grid = checkpoint.get("grid_size", online_base.grid_size)
            ckpt_rows = checkpoint.get("grid_rows", self.grid_rows)
            ckpt_cols = checkpoint.get("grid_cols", self.grid_cols)
            ckpt_dueling = checkpoint.get("dueling")
            if ckpt_dueling is None:
                keys = checkpoint["online_net"].keys()
                if any(k.startswith("head.") for k in keys):
                    ckpt_dueling = False
                else:
                    ckpt_dueling = True
            ckpt_dueling = bool(ckpt_dueling)
            needs_rebuild = (
                ckpt_hl != online_base.hidden_layers
                or ckpt_grid != online_base.grid_size
                or ckpt_rows != self.grid_rows
                or ckpt_cols != self.grid_cols
                or ckpt_dueling != online_base.dueling
            )
            if needs_rebuild:
                self.online_net = QNetwork(
                    self.obs_dim, self.n_actions, ckpt_grid, ckpt_hl,
                    ckpt_rows, ckpt_cols, ckpt_dueling,
                ).to(self.device)
                self.target_net = QNetwork(
                    self.obs_dim, self.n_actions, ckpt_grid, ckpt_hl,
                    ckpt_rows, ckpt_cols, ckpt_dueling,
                ).to(self.device)
                self.inference_net = QNetwork(
                    self.obs_dim, self.n_actions, ckpt_grid, ckpt_hl,
                    ckpt_rows, ckpt_cols, ckpt_dueling,
                ).to(self.device)
                self.online_net, self.target_net, self.inference_net = self._maybe_compile_all(
                    self.online_net, self.target_net, self.inference_net
                )
                self.optimizer = self._build_optimizer(self.optimizer.param_groups[0]["lr"])
                online_base = self._unwrap_module(self.online_net)
            self.grid_rows = ckpt_rows
            self.grid_cols = ckpt_cols
            self.dueling = ckpt_dueling
        self._load_state_dict_flexible(self.online_net, checkpoint["online_net"], "online_net")
        self._load_state_dict_flexible(self.target_net, checkpoint["target_net"], "target_net")
        self.sync_inference_net()
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint["train_steps"]
        self.epsilon = checkpoint["epsilon"]
