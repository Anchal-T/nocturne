import random
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .optimizers import build_optimizer, get_optimizer_lr
from .profiling import CudaTrainProfiler
from .noisy_layer import NoisyLinear
from .q_network import QNetwork
from .replay_buffer import ReplayBuffer


@dataclass
class DDQNAgentConfig:
    grid_size: int
    grid_channels: int
    grid_rows: int
    grid_cols: int
    device: str
    hidden_layers: Optional[List[int]] = None
    n_step: int = 3
    lr: float = 0.01
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200000
    replay_buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 1000
    dueling: bool = True
    noisy: bool = True
    mlp_depth: int = 2
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 100000
    per_epsilon: float = 1e-6
    grad_accum_steps: int = 1
    max_grad_norm: float = 10.0
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"
    inference_sync_interval: int = 4
    profile_cuda: bool = False
    profile_first_train_step: bool = True
    profile_wait_steps: int = 8
    profile_warmup_steps: int = 4
    profile_active_steps: int = 24
    profile_trace_path: str = "trace.json"
    use_muon: bool = True
    num_envs: int = 1

    @classmethod
    def from_drl_cfg(
        cls,
        drl_cfg: Dict,
        *,
        grid_size: int,
        grid_channels: int,
        grid_rows: int,
        grid_cols: int,
        device: str,
    ) -> "DDQNAgentConfig":
        return cls(
            grid_size=grid_size,
            grid_channels=grid_channels,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            device=device,
            hidden_layers=drl_cfg["hidden_layers"],
            n_step=int(drl_cfg.get("n_step", 3)),
            lr=drl_cfg["lr"],
            gamma=drl_cfg["gamma"],
            epsilon_start=drl_cfg["epsilon_start"],
            epsilon_end=drl_cfg["epsilon_end"],
            epsilon_decay_steps=drl_cfg["epsilon_decay_steps"],
            replay_buffer_size=drl_cfg["replay_buffer_size"],
            batch_size=drl_cfg["batch_size"],
            target_update_freq=drl_cfg["target_update_freq"],
            dueling=bool(drl_cfg["dueling"]),
            noisy=bool(drl_cfg["noisy"]),
            mlp_depth=int(drl_cfg["mlp_depth"]),
            alpha=drl_cfg["alpha"],
            beta_start=drl_cfg["beta_start"],
            beta_frames=drl_cfg["beta_frames"],
            per_epsilon=drl_cfg["per_epsilon"],
            grad_accum_steps=drl_cfg["grad_accum_steps"],
            max_grad_norm=drl_cfg["max_grad_norm"],
            use_torch_compile=bool(drl_cfg["use_torch_compile"]),
            compile_mode=str(drl_cfg["compile_mode"]),
            inference_sync_interval=drl_cfg["inference_sync_interval"],
            profile_cuda=bool(drl_cfg["profile_cuda"]),
            profile_first_train_step=bool(drl_cfg["profile_first_train_step"]),
            profile_wait_steps=int(drl_cfg["profile_wait_steps"]),
            profile_warmup_steps=int(drl_cfg["profile_warmup_steps"]),
            profile_active_steps=int(drl_cfg["profile_active_steps"]),
            profile_trace_path=str(drl_cfg["profile_trace_path"]),
            use_muon=bool(drl_cfg["use_muon"]),
            num_envs=int(drl_cfg["num_envs"]),
        )


class DDQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, config: DDQNAgentConfig):
        self.config = config
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(config.device)
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        self.grid_rows = config.grid_rows
        self.grid_cols = config.grid_cols
        self.grid_channels = config.grid_channels
        self.dueling = bool(config.dueling)
        self.noisy = bool(config.noisy)
        self.use_muon = bool(config.use_muon)
        self.grad_accum_steps = max(1, int(config.grad_accum_steps))
        self.max_grad_norm = float(config.max_grad_norm)
        self.inference_sync_interval = max(1, int(config.inference_sync_interval))

        self.epsilon = config.epsilon_start
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay_steps = config.epsilon_decay_steps

        self._use_amp = self.device.type == "cuda" and not self.use_muon
        self._scaler = torch.amp.GradScaler(enabled=self._use_amp)
        self.compile_mode = str(config.compile_mode)
        self._compile_enabled = (
            bool(config.use_torch_compile)
            and self.device.type == "cuda"
            and hasattr(torch, "compile")
        )
        if config.use_torch_compile and self.device.type == "cuda" and not hasattr(torch, "compile"):
            print("[DDQNAgent] torch.compile unavailable; continuing without compile.")

        self._profiler = CudaTrainProfiler(
            device=self.device,
            enabled=config.profile_cuda and config.profile_first_train_step,
            wait_steps=config.profile_wait_steps,
            warmup_steps=config.profile_warmup_steps,
            active_steps=config.profile_active_steps,
            trace_path=config.profile_trace_path,
        )
        self._profile_cuda = config.profile_cuda

        self.online_net, self.target_net, self.inference_net = self._build_networks(config)
        self.inference_lock = threading.Lock()
        self.sync_target()
        self.sync_inference_net()

        self.optimizer = build_optimizer(
            self.online_net, config.lr, self.device.type, self.use_muon,
        )

        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim,
            size=config.replay_buffer_size,
            batch_size=config.batch_size,
            n_step=config.n_step,
            gamma=config.gamma,
            alpha=config.alpha,
            beta_start=config.beta_start,
            beta_frames=config.beta_frames,
            epsilon=config.per_epsilon,
            num_envs=config.num_envs,
        )
        self.train_steps = 0
        self._grad_accum_counter = 0
        self._pin_buffers = self._init_pin_buffers(config.batch_size, obs_dim)

    def _build_networks(self, config: DDQNAgentConfig):
        def _make_net() -> QNetwork:
            return QNetwork(
                obs_dim=self.obs_dim,
                n_actions=self.n_actions,
                grid_size=config.grid_size,
                hidden_layers=config.hidden_layers,
                grid_channels=config.grid_channels,
                grid_rows=config.grid_rows,
                grid_cols=config.grid_cols,
                dueling=self.dueling,
                noisy=self.noisy,
                mlp_depth=config.mlp_depth,
            ).to(self.device)

        online = _make_net()
        target = _make_net()
        inference = _make_net()
        online, target, inference = self._maybe_compile(online, target, inference)
        self._set_eval_network_modes(target, inference)
        return online, target, inference

    def _init_pin_buffers(self, batch_size: int, obs_dim: int) -> Optional[dict]:
        if self.device.type != "cuda":
            return None
        return {
            "obs": torch.zeros(batch_size, obs_dim, pin_memory=True),
            "next_obs": torch.zeros(batch_size, obs_dim, pin_memory=True),
            "acts": torch.zeros(batch_size, dtype=torch.long, pin_memory=True),
            "rews": torch.zeros(batch_size, pin_memory=True),
            "dones": torch.zeros(batch_size, pin_memory=True),
            "weights": torch.zeros(batch_size, pin_memory=True),
        }

    def _maybe_compile(
        self, online: nn.Module, target: nn.Module, inference: nn.Module,
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        if not self._compile_enabled:
            return online, target, inference
        try:
            return (
                torch.compile(online, mode=self.compile_mode),
                torch.compile(target, mode=self.compile_mode),
                inference,  # keep eager — async actor path can conflict with dynamo
            )
        except Exception as exc:
            print(f"[DDQNAgent] torch.compile failed ({exc}); continuing in eager mode.")
            self._compile_enabled = False
            return online, target, inference

    def _set_eval_network_modes(self, target: nn.Module, inference: nn.Module) -> None:
        target.eval()
        inference.eval()
        if self.noisy:
            self._set_noisy_layers_training(inference, True)

    def _reset_inference_noise_if_needed(self) -> None:
        if self.noisy:
            self._unwrap_module(self.inference_net).reset_noise()

    @staticmethod
    def _set_noisy_layers_training(module: nn.Module, training: bool) -> None:
        for submodule in DDQNAgent._unwrap_module(module).modules():
            if isinstance(submodule, NoisyLinear):
                submodule.train(training)

    # --- Action Selection ---

    def select_action(self, state: np.ndarray) -> int:
        if not self.noisy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with self.inference_lock:
                self._reset_inference_noise_if_needed()
                q_values = self.inference_net(state_t)
            return q_values.argmax(dim=1).item()

    def select_action_batch(self, states: np.ndarray) -> np.ndarray:
        batch_size = states.shape[0]
        actions = np.zeros(batch_size, dtype=int)

        if self.noisy:
            # NoisyNet handles exploration — always greedy
            with torch.no_grad():
                states_t = torch.FloatTensor(states).to(self.device)
                with self.inference_lock:
                    self._reset_inference_noise_if_needed()
                    q_values = self.inference_net(states_t)
                actions[:] = q_values.argmax(dim=1).cpu().numpy()
            return actions

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

    # --- Replay Buffer ---

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store(
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )

    def store_transition_batch(self, states, actions, rewards, next_states, dones, env_ids=None):
        self.replay_buffer.store_batch(states, actions, rewards, next_states, dones, env_ids=env_ids)

    # --- Training ---

    def _batch_to_tensors(self, batch: Dict[str, np.ndarray]):
        if self._pin_buffers is not None:
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
            return (
                pb["obs"].to(self.device, non_blocking=True),
                pb["next_obs"].to(self.device, non_blocking=True),
                pb["acts"].to(self.device, non_blocking=True),
                pb["rews"].to(self.device, non_blocking=True),
                pb["dones"].to(self.device, non_blocking=True),
                pb["weights"].to(self.device, non_blocking=True),
            )

        return (
            torch.FloatTensor(batch["obs"]).to(self.device),
            torch.FloatTensor(batch["next_obs"]).to(self.device),
            torch.LongTensor(batch["acts"]).to(self.device),
            torch.FloatTensor(batch["rews"]).to(self.device),
            torch.FloatTensor(batch["done"]).to(self.device),
            torch.FloatTensor(batch["weights"]).to(self.device),
        )

    def _compute_loss(self, obs_t, acts_t, next_obs_t, rews_t, dones_t, weights_t):
        with torch.amp.autocast(device_type=self.device.type, enabled=self._use_amp):
            current_q = self.online_net(obs_t).gather(1, acts_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                best_acts = self.online_net(next_obs_t).argmax(dim=1)
                next_q = self.target_net(next_obs_t).gather(1, best_acts.unsqueeze(1)).squeeze(1)
                target_q = rews_t + (self.gamma ** self.config.n_step) * next_q * (1.0 - dones_t)
            td_errors = current_q - target_q
            loss = (weights_t * F.huber_loss(current_q, target_q, reduction='none')).mean()
        return loss, td_errors

    def _apply_gradient_step(self, loss: torch.Tensor) -> None:
        scale = 1.0 / self.grad_accum_steps
        if self._use_amp:
            self._scaler.scale(loss * scale).backward()
        else:
            (loss * scale).backward()
        self._grad_accum_counter += 1

        if self._grad_accum_counter < self.grad_accum_steps:
            return

        if self._use_amp:
            self._scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.max_grad_norm)
        if self._use_amp:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._grad_accum_counter = 0

    def train_step(self, env_steps: int = 0) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        self._profiler.maybe_start()

        # Reset noisy layers before each training forward pass
        self.online_net.train()
        if self.noisy:
            self._unwrap_module(self.online_net).reset_noise()
            self._unwrap_module(self.target_net).reset_noise()

        batch = self.replay_buffer.sample_batch()
        obs_t, next_obs_t, acts_t, rews_t, dones_t, weights_t = self._batch_to_tensors(batch)
        loss, td_errors = self._compute_loss(obs_t, acts_t, next_obs_t, rews_t, dones_t, weights_t)
        self._apply_gradient_step(loss)

        # Post-step bookkeeping
        with torch.no_grad():
            new_priorities = td_errors.abs().cpu().numpy()
        self.replay_buffer.update_priorities(batch["indices"], new_priorities)
        self.train_steps += 1
        self.replay_buffer.update_beta(env_steps)
        self._decay_epsilon(env_steps)

        should_sync_inference = (self.train_steps % self.inference_sync_interval) == 0
        if self.train_steps % self.target_update_freq == 0:
            self.sync_target()
            should_sync_inference = True
        if should_sync_inference:
            self.sync_inference_net()

        self._profiler.maybe_log_memory(self.train_steps)
        self._profiler.advance()

        return float(loss.item())

    # --- Network Sync ---

    def sync_target(self):
        online_state = self._unwrap_module(self.online_net).state_dict()
        self._load_state_dict_flexible(self.target_net, online_state, "target_sync")

    def sync_inference_net(self):
        with self.inference_lock:
            online_state = self._unwrap_module(self.online_net).state_dict()
            self._load_state_dict_flexible(self.inference_net, online_state, "inference_sync")

    def export_inference_state(self, refresh: bool = False) -> Dict[str, Any]:
        if refresh:
            self.sync_inference_net()

        with self.inference_lock:
            inference_state = self._unwrap_module(self.inference_net).state_dict()
            return {
                "inference_net": self._state_dict_to_cpu(inference_state),
                "train_steps": self.train_steps,
                "epsilon": self.epsilon,
            }

    def load_inference_state(self, state: Dict[str, Any]) -> None:
        inference_state = state.get("inference_net")
        if inference_state is None:
            raise KeyError("Inference state payload is missing 'inference_net'.")

        with self.inference_lock:
            self._load_state_dict_flexible(self.inference_net, inference_state, "inference_net")

        if "train_steps" in state:
            self.train_steps = int(state["train_steps"])
        if "epsilon" in state:
            self.epsilon = float(state["epsilon"])

    def update_exploration(self, env_steps: int) -> float:
        self._decay_epsilon(env_steps)
        return self.epsilon

    def _decay_epsilon(self, env_steps: int):
        if env_steps <= 0:
            return
        fraction = min(1.0, env_steps / max(1, self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def finalize_profiling(self) -> None:
        self._profiler.finalize(export_trace=True)

    # --- Save / Load ---

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
                "grid_channels": self.grid_channels,
                "grid_rows": self.grid_rows,
                "grid_cols": self.grid_cols,
                "dueling": online_base.dueling,
                "noisy": getattr(online_base, "noisy", True),
                "mlp_depth": int(getattr(online_base, "mlp_depth", self.config.mlp_depth)),
                "use_muon": self.use_muon,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if "use_muon" in checkpoint:
            ckpt_use_muon = bool(checkpoint["use_muon"])
            if ckpt_use_muon != self.use_muon:
                self.use_muon = ckpt_use_muon
                lr = get_optimizer_lr(self.optimizer)
                self.optimizer = build_optimizer(
                    self.online_net, lr, self.device.type, self.use_muon,
                )
                self._use_amp = self.device.type == "cuda" and not self.use_muon
                self._scaler = torch.amp.GradScaler(enabled=self._use_amp)

        self._load_checkpoint_architecture(checkpoint)
        self._load_state_dict_flexible(self.online_net, checkpoint["online_net"], "online_net")
        self._load_state_dict_flexible(self.target_net, checkpoint["target_net"], "target_net")
        self.sync_inference_net()
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception:
            pass
        self.train_steps = checkpoint["train_steps"]
        self.epsilon = checkpoint["epsilon"]

    def _load_checkpoint_architecture(self, checkpoint: Dict[str, Any]) -> None:
        online_base = self._unwrap_module(self.online_net)
        required_keys = ("hidden_layers", "grid_size", "grid_rows", "grid_cols", "dueling", "noisy")
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise KeyError(
                "Checkpoint is missing required architecture fields: "
                f"{', '.join(missing_keys)}."
            )

        ckpt_hl = checkpoint["hidden_layers"]
        ckpt_grid = checkpoint["grid_size"]
        ckpt_rows = checkpoint["grid_rows"]
        ckpt_cols = checkpoint["grid_cols"]
        ckpt_channels = int(
            checkpoint.get(
                "grid_channels",
                max(1, ckpt_grid // max(1, ckpt_rows * ckpt_cols)),
            )
        )
        ckpt_dueling = bool(checkpoint["dueling"])
        ckpt_noisy = bool(checkpoint["noisy"])
        ckpt_mlp_depth = int(
            checkpoint.get(
                "mlp_depth",
                self._infer_mlp_depth_from_state_dict(checkpoint.get("online_net")),
            )
        )

        needs_rebuild = (
            ckpt_hl != online_base.hidden_layers
            or ckpt_grid != online_base.grid_size
            or ckpt_channels != getattr(online_base, "grid_channels", 1)
            or ckpt_rows != self.grid_rows
            or ckpt_cols != self.grid_cols
            or ckpt_dueling != online_base.dueling
            or ckpt_noisy != getattr(online_base, "noisy", True)
            or ckpt_mlp_depth != getattr(online_base, "mlp_depth", 2)
        )

        if needs_rebuild:
            self._rebuild_networks(
                ckpt_grid,
                ckpt_channels,
                ckpt_hl,
                ckpt_rows,
                ckpt_cols,
                ckpt_dueling,
                ckpt_noisy,
                ckpt_mlp_depth,
            )

        self.grid_channels = ckpt_channels
        self.grid_rows = ckpt_rows
        self.grid_cols = ckpt_cols
        self.dueling = ckpt_dueling
        self.noisy = ckpt_noisy
        self.config.mlp_depth = ckpt_mlp_depth

    def _rebuild_networks(
        self, grid_size, grid_channels, hidden_layers, grid_rows, grid_cols, dueling, noisy, mlp_depth,
    ):
        def _make(d, n, md):
            return QNetwork(
                self.obs_dim, self.n_actions, grid_size, hidden_layers,
                grid_channels, grid_rows, grid_cols, d, noisy=n, mlp_depth=md,
            ).to(self.device)

        self.online_net = _make(dueling, noisy, mlp_depth)
        self.target_net = _make(dueling, noisy, mlp_depth)
        self.inference_net = _make(dueling, noisy, mlp_depth)
        self.online_net, self.target_net, self.inference_net = self._maybe_compile(
            self.online_net, self.target_net, self.inference_net,
        )
        self._set_eval_network_modes(self.target_net, self.inference_net)
        lr = get_optimizer_lr(self.optimizer)
        self.optimizer = build_optimizer(self.online_net, lr, self.device.type, self.use_muon)

    # --- State Dict Utilities ---

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
    def _infer_mlp_depth_from_state_dict(state_dict: Optional[Dict[str, torch.Tensor]]) -> int:
        if not state_dict:
            return 2

        keys = DDQNAgent._strip_known_prefixes(state_dict).keys()
        max_block_idx = -1
        for key in keys:
            parts = key.split(".")
            if len(parts) < 3:
                continue
            if parts[0] not in {"advantage_head", "value_head", "head"}:
                continue
            if parts[1] != "residual_blocks":
                continue
            try:
                max_block_idx = max(max_block_idx, int(parts[2]))
            except ValueError:
                continue

        if max_block_idx >= 0:
            # ResidualMLP uses floor(mlp_depth / 4) residual blocks.
            return (max_block_idx + 1) * 4

        # Depths 0..3 map to the same architecture; prefer legacy default.
        return 2

    @staticmethod
    def _state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in state_dict.items()
        }

    def _load_state_dict_flexible(self, module: nn.Module, state_dict: Dict[str, torch.Tensor], name: str) -> None:
        base_module = self._unwrap_module(module)
        normalized_state = self._strip_known_prefixes(state_dict)
        try:
            base_module.load_state_dict(normalized_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to load {name} state dict across known key formats: {exc}")
