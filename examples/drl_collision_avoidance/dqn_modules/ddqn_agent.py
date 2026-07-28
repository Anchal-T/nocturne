import logging
import random
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .noisy_layer import NoisyLinear
from .optimizers import build_optimizer, get_optimizer_lr
from .profiling import CudaTrainProfiler
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
    use_cuda_graph: bool = False
    cuda_graph_warmup_steps: int = 3
    inference_sync_interval: int = 4
    profile_cuda: bool = False
    profile_first_train_step: bool = True
    profile_wait_steps: int = 8
    profile_warmup_steps: int = 4
    profile_active_steps: int = 24
    profile_trace_path: str = "trace.json"
    use_muon: bool = True
    num_envs: int = 1
    # Shrink-and-perturb reset (SR-SPR / BBF style) to protect plasticity
    # under high replay ratios. 0 disables.
    spr_interval: int = 0
    spr_scale: float = 0.5
    spr_noise_std: float = 0.001
    # GPU-resident replay: store obs once, derive next_obs by index.
    gpu_replay: bool = True
    replay_obs_dtype: str = "float16"
    # Sync loss.item() / priority D2H only every N train steps.
    loss_sync_interval: int = 32

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
            use_cuda_graph=bool(drl_cfg.get("use_cuda_graph", False)),
            cuda_graph_warmup_steps=int(drl_cfg.get("cuda_graph_warmup_steps", 3)),
            inference_sync_interval=drl_cfg["inference_sync_interval"],
            profile_cuda=bool(drl_cfg["profile_cuda"]),
            profile_first_train_step=bool(drl_cfg["profile_first_train_step"]),
            profile_wait_steps=int(drl_cfg["profile_wait_steps"]),
            profile_warmup_steps=int(drl_cfg["profile_warmup_steps"]),
            profile_active_steps=int(drl_cfg["profile_active_steps"]),
            profile_trace_path=str(drl_cfg["profile_trace_path"]),
            use_muon=bool(drl_cfg["use_muon"]),
            num_envs=int(drl_cfg["num_envs"]),
            spr_interval=int(drl_cfg.get("spr_interval", 0)),
            spr_scale=float(drl_cfg.get("spr_scale", 0.5)),
            spr_noise_std=float(drl_cfg.get("spr_noise_std", 0.001)),
            gpu_replay=bool(drl_cfg.get("gpu_replay", True)),
            replay_obs_dtype=str(drl_cfg.get("replay_obs_dtype", "float16")),
            loss_sync_interval=int(drl_cfg.get("loss_sync_interval", 32)),
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

        # Shrink-and-perturb reset (SR-SPR / BBF style) to protect plasticity
        # under high replay ratios. Every spr_interval gradient steps the
        # online net is reset to a shrunken copy of itself with noise injected.
        self.spr_interval = int(config.spr_interval)
        self.spr_scale = float(config.spr_scale)
        self.spr_noise_std = float(config.spr_noise_std)

        # Must be set before _reset_amp_state (bf16 vs fp16 + GradScaler).
        self._use_cuda_graph = (
            config.use_cuda_graph and self.device.type == "cuda"
        )
        self._cuda_graph_warmup_steps = config.cuda_graph_warmup_steps

        self._reset_amp_state()
        self.compile_mode = str(config.compile_mode)
        self._compile_enabled = (
            bool(config.use_torch_compile)
            and self.device.type == "cuda"
            and hasattr(torch, "compile")
        )
        if (
            config.use_torch_compile
            and self.device.type == "cuda"
            and not hasattr(torch, "compile")
        ):
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

        self.online_net, self.target_net, self.inference_net = self._build_networks(
            config
        )
        self.inference_lock = threading.Lock()
        self.sync_target()
        self.sync_inference_net()

        self.optimizer = build_optimizer(
            self.online_net,
            config.lr,
            self.device.type,
            self.use_muon,
        )

        replay_device = (
            self.device
            if (config.gpu_replay and self.device.type == "cuda")
            else None
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
            device=replay_device,
            obs_dtype=config.replay_obs_dtype,
        )
        self._gpu_replay = bool(replay_device is not None)
        self._gamma_n: float = self.gamma**config.n_step
        self.train_steps = 0
        self._grad_accum_counter = 0
        self._pin_buffers = self._init_pin_buffers(config.batch_size, obs_dim)
        self._inference_pin_buf = self._init_inference_pin_buf(config.num_envs, obs_dim)

        # CUDA graph runtime state (flag already set above for AMP).
        self._train_graph = None
        self._train_graph_inputs = None
        self._train_graph_loss = None
        self._train_graph_td_errors = None
        self._inference_graph = None
        self._inference_graph_input = None
        self._inference_graph_output = None
        self._inference_graph_max_batch = max(1, int(config.num_envs))
        self._inference_cuda_graph = self._use_cuda_graph

        # Separate streams so actor inference and learner training can overlap
        # when they share one GPU (same process, or via the GPU scheduler).
        self._train_stream = (
            torch.cuda.Stream() if self.device.type == "cuda" else None
        )
        self._infer_stream = (
            torch.cuda.Stream() if self.device.type == "cuda" else None
        )

        # Deferred host sync for loss / PER priorities.
        self._loss_sync_interval = max(1, int(config.loss_sync_interval))
        self._loss_accum = None
        self._loss_count = 0
        self._last_loss: Optional[float] = None
        self._pending_prio: List[Tuple[np.ndarray, torch.Tensor]] = []
        # Ring of pinned staging buffers for async |td_error| D2H.
        self._td_pin_ring: List[torch.Tensor] = []
        self._td_pin_slot = 0
        if self.device.type == "cuda":
            ring = max(self._loss_sync_interval, 4)
            self._td_pin_ring = [
                torch.empty(config.batch_size, pin_memory=True) for _ in range(ring)
            ]

        # TF32 matmul precision boost on Ampere+.
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _instantiate_networks(
        self, make_fn: Callable[[], QNetwork]
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """Create three networks via make_fn, apply compile and eval modes."""
        online, target, inference = make_fn(), make_fn(), make_fn()
        online, target, inference = self._maybe_compile(online, target, inference)
        self._set_eval_network_modes(target, inference)
        return online, target, inference

    def _build_networks(self, config: DDQNAgentConfig):
        def _make() -> QNetwork:
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

        return self._instantiate_networks(_make)

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

    def _init_inference_pin_buf(
        self, max_batch: int, obs_dim: int
    ) -> Optional[torch.Tensor]:
        if self.device.type != "cuda":
            return None
        return torch.zeros(max_batch, obs_dim, pin_memory=True)

    def _maybe_compile(
        self,
        online: nn.Module,
        target: nn.Module,
        inference: nn.Module,
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
            print(
                f"[DDQNAgent] torch.compile failed ({exc}); continuing in eager mode."
            )
            self._compile_enabled = False
            return online, target, inference

    def _reset_amp_state(self) -> None:
        # autocast (bf16/fp16 forward) is compatible with Muon; only GradScaler
        # (FP16 loss scaling) conflicts because Muon updates from unscaled grads.
        self._use_amp = self.device.type == "cuda"
        # CUDA graphs require bf16 autocast (no GradScaler) because the scaler's
        # inf/nan checks and dynamic scale updates are host-side logic that
        # cannot be captured or replayed.
        if self._use_cuda_graph:
            self._amp_dtype = torch.bfloat16
            self._use_scaler = False
        else:
            self._amp_dtype = torch.float16
            self._use_scaler = self._use_amp and not self.use_muon
        self._scaler = torch.amp.GradScaler(enabled=self._use_scaler)

    def _set_eval_network_modes(self, target: nn.Module, inference: nn.Module) -> None:
        target.eval()
        inference.eval()
        if self.noisy:
            self._set_noisy_layers_training(inference, True)

    def _reset_inference_noise_if_needed(self) -> None:
        if self.noisy:
            self._unwrap_module(self.inference_net).reset_noise()

    def _capture_inference_graph(self):
        """Capture the inference net forward pass as a CUDA graph.

        Uses a fixed-size static input tensor (padded to num_envs). The QNetwork
        processes samples independently, so padding does not affect the first
        n rows of the output.
        """
        max_batch = self._inference_graph_max_batch
        self._inference_graph_input = torch.zeros(
            max_batch, self.obs_dim, device=self.device, dtype=torch.float32
        )
        # Warmup
        for _ in range(3):
            with self.inference_lock:
                self._reset_inference_noise_if_needed()
                _ = self.inference_net(self._inference_graph_input)
        # Capture
        self._inference_graph = torch.cuda.CUDAGraph()
        with self.inference_lock:
            self._reset_inference_noise_if_needed()
            with torch.cuda.graph(self._inference_graph):
                self._inference_graph_output = self.inference_net(
                    self._inference_graph_input
                )

    def _greedy_inference(self, states_t: torch.Tensor) -> torch.Tensor:
        """Forward pass on inference_net under lock with noise reset. Returns Q-values."""
        with self.inference_lock:
            self._reset_inference_noise_if_needed()
            return self.inference_net(states_t)

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
            state_t = (
                torch.from_numpy(np.asarray(state, dtype=np.float32))
                .unsqueeze(0)
                .to(self.device)
            )
            return self._greedy_inference(state_t).argmax(dim=1).item()

    def select_action_batch(self, states: np.ndarray) -> np.ndarray:
        batch_size = states.shape[0]
        actions = np.zeros(batch_size, dtype=int)

        if self.noisy:
            # NoisyNet handles exploration — always greedy
            with torch.no_grad():
                states_t = self._to_device_pinned(states)
                q_values = self._greedy_inference_graphed(states_t, batch_size)
                actions[:] = q_values.argmax(dim=1).cpu().numpy()
            return actions

        random_mask = np.random.rand(batch_size) < self.epsilon
        if random_mask.any():
            actions[random_mask] = np.random.randint(
                0, self.n_actions, size=random_mask.sum()
            )

        greedy_mask = ~random_mask
        if greedy_mask.any():
            with torch.no_grad():
                states_t = self._to_device_pinned(states[greedy_mask])
                n_greedy = int(greedy_mask.sum())
                q_values = self._greedy_inference_graphed(states_t, n_greedy)
                actions[greedy_mask] = q_values.argmax(dim=1).cpu().numpy()
        return actions

    def _greedy_inference_graphed(
        self, states_t: torch.Tensor, n_valid: int
    ) -> torch.Tensor:
        """Run inference via CUDA graph if available and shapes fit, else eager."""
        stream_ctx = (
            torch.cuda.stream(self._infer_stream)
            if self._infer_stream is not None
            else nullcontext()
        )
        with stream_ctx:
            use_graph = (
                self._inference_cuda_graph
                and n_valid <= self._inference_graph_max_batch
                and self.device.type == "cuda"
            )
            if use_graph:
                try:
                    if self._inference_graph is None:
                        self._capture_inference_graph()
                    # Copy into padded static tensor and replay.
                    self._inference_graph_input[:n_valid].copy_(states_t)
                    if n_valid < self._inference_graph_max_batch:
                        self._inference_graph_input[n_valid:].zero_()
                    with self.inference_lock:
                        self._reset_inference_noise_if_needed()
                        self._inference_graph.replay()
                    return self._inference_graph_output[:n_valid]
                except Exception as exc:
                    print(
                        f"[DDQNAgent] inference CUDA graph failed ({exc}); "
                        "falling back to eager inference."
                    )
                    self._inference_graph = None
                    self._inference_cuda_graph = False
            # Eager fallback
            with self.inference_lock:
                self._reset_inference_noise_if_needed()
                return self.inference_net(states_t)

    def _to_device_pinned(self, np_array: np.ndarray) -> torch.Tensor:
        """Copy numpy array to GPU via pinned memory for faster H2D transfer."""
        n = np_array.shape[0]
        if (
            self._inference_pin_buf is not None
            and n <= self._inference_pin_buf.shape[0]
        ):
            self._inference_pin_buf[:n].copy_(torch.from_numpy(np_array))
            return self._inference_pin_buf[:n].to(self.device, non_blocking=False)
        return torch.from_numpy(np.asarray(np_array, dtype=np.float32)).to(self.device)

    # --- Replay Buffer ---

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store(
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )

    def store_transition_batch(
        self, states, actions, rewards, next_states, dones, env_ids=None
    ):
        self.replay_buffer.store_batch(
            states, actions, rewards, next_states, dones, env_ids=env_ids
        )

    # --- Training ---

    def _batch_to_tensors(self, batch):
        # GPU-resident replay already returns device tensors — no H2D copy.
        if self._gpu_replay and torch.is_tensor(batch["obs"]):
            obs = batch["obs"].float()
            next_obs = batch["next_obs"].float()
            return (
                obs,
                next_obs,
                batch["acts"],
                batch["rews"],
                batch["done"],
                batch["weights"],
            )

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
            torch.from_numpy(batch["obs"]).to(self.device),
            torch.from_numpy(batch["next_obs"]).to(self.device),
            torch.from_numpy(batch["acts"].astype(np.int64, copy=False)).to(
                self.device
            ),
            torch.from_numpy(batch["rews"]).to(self.device),
            torch.from_numpy(batch["done"]).to(self.device),
            torch.from_numpy(batch["weights"]).to(self.device),
        )

    def _compute_loss(self, obs_t, acts_t, next_obs_t, rews_t, dones_t, weights_t):
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self._use_amp,
        ):
            current_q = self.online_net(obs_t).gather(1, acts_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                best_acts = self.online_net(next_obs_t).argmax(dim=1)
                next_q = (
                    self.target_net(next_obs_t)
                    .gather(1, best_acts.unsqueeze(1))
                    .squeeze(1)
                )
                target_q = rews_t + self._gamma_n * next_q * (1.0 - dones_t)
            td_errors = current_q - target_q
            loss = (
                weights_t * F.huber_loss(current_q, target_q, reduction="none")
            ).mean()
        return loss, td_errors

    def _apply_gradient_step(self, loss: torch.Tensor) -> None:
        scale = 1.0 / self.grad_accum_steps
        if self._use_scaler:
            self._scaler.scale(loss * scale).backward()
        else:
            (loss * scale).backward()
        self._grad_accum_counter += 1

        if self._grad_accum_counter < self.grad_accum_steps:
            return

        if self._use_scaler:
            self._scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(
            self.online_net.parameters(), max_norm=self.max_grad_norm
        )
        if self._use_scaler:
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

        stream_ctx = (
            torch.cuda.stream(self._train_stream)
            if self._train_stream is not None
            else nullcontext()
        )
        with stream_ctx:
            # Noisy layers re-sample each forward pass; reset ensures fresh noise.
            self.online_net.train()
            if self.noisy:
                self._unwrap_module(self.online_net).reset_noise()
                self._unwrap_module(self.target_net).reset_noise()

            batch = self.replay_buffer.sample_batch()
            obs_t, next_obs_t, acts_t, rews_t, dones_t, weights_t = (
                self._batch_to_tensors(batch)
            )

            # Use CUDA graph replay after warmup if enabled and shapes are stable.
            if (
                self._use_cuda_graph
                and self.train_steps >= self._cuda_graph_warmup_steps
                and self.grad_accum_steps == 1
            ):
                if self._train_graph is None:
                    self._capture_train_graph(
                        obs_t, next_obs_t, acts_t, rews_t, dones_t, weights_t
                    )
                loss, td_errors = self._replay_train_graph(
                    obs_t, next_obs_t, acts_t, rews_t, dones_t, weights_t
                )
            else:
                loss, td_errors = self._compute_loss(
                    obs_t, acts_t, next_obs_t, rews_t, dones_t, weights_t
                )
                self._apply_gradient_step(loss)

            # Queue priority update without forcing a D2H sync this step.
            self._queue_priority_update(batch["indices"], td_errors)

            # Accumulate loss on-device; sync to host every N steps.
            loss_det = loss.detach()
            if self._loss_accum is None:
                self._loss_accum = loss_det
            else:
                self._loss_accum = self._loss_accum + loss_det
            self._loss_count += 1

        self.train_steps += 1
        self.replay_buffer.update_beta(env_steps)
        self.update_exploration(env_steps)

        if self.train_steps % self._loss_sync_interval == 0:
            self._flush_host_syncs()

        # Shrink-and-perturb reset to protect plasticity under high replay ratio.
        if (self.spr_interval > 0
                and self.train_steps % self.spr_interval == 0
                and self.train_steps > 0):
            self._flush_host_syncs()
            self._shrink_and_perturb()

        should_sync_inference = (self.train_steps % self.inference_sync_interval) == 0
        if self.train_steps % self.target_update_freq == 0:
            self.sync_target()
            should_sync_inference = True
        if should_sync_inference:
            self.sync_inference_net()

        self._profiler.maybe_log_memory(self.train_steps)
        self._profiler.advance()

        return self._last_loss

    def _queue_priority_update(self, indices, td_errors: torch.Tensor) -> None:
        """Stage a non-blocking D2H of |td_errors| for a later priority update."""
        td_abs = td_errors.detach().abs()
        idx_np = (
            indices.detach().cpu().numpy()
            if torch.is_tensor(indices)
            else np.asarray(indices, dtype=np.int64).copy()
        )
        if (
            self._td_pin_ring
            and td_abs.shape[0] == self._td_pin_ring[0].shape[0]
        ):
            staged = self._td_pin_ring[self._td_pin_slot]
            self._td_pin_slot = (self._td_pin_slot + 1) % len(self._td_pin_ring)
            staged.copy_(td_abs, non_blocking=True)
            self._pending_prio.append((idx_np, staged))
        else:
            self._pending_prio.append((idx_np, td_abs.detach().cpu()))

    def _flush_host_syncs(self) -> None:
        """Sync pending loss and PER priority updates to the host."""
        if self._loss_count > 0 and self._loss_accum is not None:
            self._last_loss = float(
                (self._loss_accum / self._loss_count).item()
            )
            self._loss_accum = None
            self._loss_count = 0

        if self._pending_prio:
            # Ensure all non-blocking D2H copies have completed.
            if self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()
            for indices, td_tensor in self._pending_prio:
                self.replay_buffer.update_priorities(
                    indices, td_tensor.numpy()
                )
            self._pending_prio.clear()

    # --- CUDA Graph Capture ---

    def _capture_train_graph(self, obs_t, next_obs_t, acts_t, rews_t, dones_t, weights_t):
        """Capture forward + backward + optimizer as a single CUDA graph.

        Static input tensors are allocated once and reused for every replay.
        The graph reads from these tensors, so the caller must copy new batch
        data into them before each replay.
        """
        # Allocate static input tensors on the GPU.
        self._train_graph_inputs = {
            "obs": torch.zeros_like(obs_t),
            "next_obs": torch.zeros_like(next_obs_t),
            "acts": torch.zeros_like(acts_t),
            "rews": torch.zeros_like(rews_t),
            "dones": torch.zeros_like(dones_t),
            "weights": torch.zeros_like(weights_t),
        }
        # Copy current batch into static inputs so warmup runs on real data.
        self._train_graph_inputs["obs"].copy_(obs_t)
        self._train_graph_inputs["next_obs"].copy_(next_obs_t)
        self._train_graph_inputs["acts"].copy_(acts_t)
        self._train_graph_inputs["rews"].copy_(rews_t)
        self._train_graph_inputs["dones"].copy_(dones_t)
        self._train_graph_inputs["weights"].copy_(weights_t)

        # Warmup runs (required before graph capture for cudnn/cublas init).
        for _ in range(3):
            self._run_graph_interior()

        # Capture the graph.
        self._train_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._train_graph):
            self._train_graph_loss, self._train_graph_td_errors = (
                self._run_graph_interior()
            )

    def _run_graph_interior(self):
        """Forward + backward + optimizer step using static input tensors.

        Returns (loss, td_errors) from the static output tensors.
        """
        gi = self._train_graph_inputs
        loss, td_errors = self._compute_loss(
            gi["obs"], gi["acts"], gi["next_obs"], gi["rews"], gi["dones"], gi["weights"]
        )
        self._apply_gradient_step(loss)
        return loss, td_errors

    def _replay_train_graph(self, obs_t, next_obs_t, acts_t, rews_t, dones_t, weights_t):
        """Copy new inputs into static tensors and replay the captured graph."""
        gi = self._train_graph_inputs
        gi["obs"].copy_(obs_t)
        gi["next_obs"].copy_(next_obs_t)
        gi["acts"].copy_(acts_t)
        gi["rews"].copy_(rews_t)
        gi["dones"].copy_(dones_t)
        gi["weights"].copy_(weights_t)
        self._train_graph.replay()
        return self._train_graph_loss, self._train_graph_td_errors

    # --- Network Sync ---

    def _shrink_and_perturb(self):
        """Shrink-and-perturb reset (SR-SPR / BBF style).

        Resets the online network to a shrunken copy of itself (scaled by
        ``spr_scale``) with Gaussian noise injected (``spr_noise_std``). This
        restores plasticity lost under high replay ratios while preserving
        the learned features. The optimizer state is also reset.
        """
        online = self._unwrap_module(self.online_net)
        with torch.no_grad():
            for param in online.parameters():
                param.mul_(self.spr_scale)
                if self.spr_noise_std > 0.0:
                    param.add_(torch.randn_like(param) * self.spr_noise_std)
        # Reset the optimizer so it doesn't apply stale momentum to the
        # perturbed weights.
        self.optimizer.zero_grad(set_to_none=True)
        if hasattr(self.optimizer, 'reset_state'):
            self.optimizer.reset_state()
        else:
            # Fallback: rebuild the optimizer from scratch with the same lr.
            lr = get_optimizer_lr(self.optimizer)
            self.optimizer = build_optimizer(
                self._unwrap_module(self.online_net),
                lr,
                self.device.type,
                self.use_muon,
            )
        # Invalidate the CUDA graph since the optimizer was rebuilt and its
        # momentum buffers now live at different addresses.
        self._train_graph = None
        self._train_graph_inputs = None

    def sync_target(self):
        online_state = self._unwrap_module(self.online_net).state_dict()
        self._load_state_dict_flexible(self.target_net, online_state, "target_sync")

    def sync_inference_net(self):
        with self.inference_lock:
            online_state = self._unwrap_module(self.online_net).state_dict()
            self._load_state_dict_flexible(
                self.inference_net, online_state, "inference_sync"
            )

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
            self._load_state_dict_flexible(
                self.inference_net, inference_state, "inference_net"
            )

        if "train_steps" in state:
            self.train_steps = int(state["train_steps"])
        if "epsilon" in state:
            self.epsilon = float(state["epsilon"])

    def update_exploration(self, env_steps: int) -> float:
        # At env_steps == 0, epsilon is left unchanged: it holds epsilon_start
        # on first call, or the last value restored from a checkpoint.
        if env_steps > 0:
            fraction = min(1.0, env_steps / max(1, self.epsilon_decay_steps))
            self.epsilon = self.epsilon_start + fraction * (
                self.epsilon_end - self.epsilon_start
            )
        return self.epsilon

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
                "obs_dim": self.obs_dim,
                "hidden_layers": online_base.hidden_layers,
                "grid_size": online_base.grid_size,
                "grid_channels": self.grid_channels,
                "grid_rows": self.grid_rows,
                "grid_cols": self.grid_cols,
                "dueling": online_base.dueling,
                "noisy": online_base.noisy,
                "mlp_depth": int(online_base.mlp_depth),
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
                    self.online_net,
                    lr,
                    self.device.type,
                    self.use_muon,
                )
                self._reset_amp_state()

        self._load_checkpoint_architecture(checkpoint)
        self._load_state_dict_flexible(
            self.online_net, checkpoint["online_net"], "online_net"
        )
        self._load_state_dict_flexible(
            self.target_net, checkpoint["target_net"], "target_net"
        )
        self.sync_inference_net()
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as exc:
            logging.warning(
                f"Failed to load optimizer state from checkpoint: {exc}. "
                "Continuing with freshly initialized optimizer."
            )
        self.train_steps = checkpoint["train_steps"]
        self.epsilon = checkpoint["epsilon"]

    def _load_checkpoint_architecture(self, checkpoint: Dict[str, Any]) -> None:
        online_base = self._unwrap_module(self.online_net)
        required_keys = (
            "hidden_layers",
            "grid_size",
            "grid_rows",
            "grid_cols",
            "dueling",
            "noisy",
        )
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
            or ckpt_channels != online_base.grid_channels
            or ckpt_rows != self.grid_rows
            or ckpt_cols != self.grid_cols
            or ckpt_dueling != online_base.dueling
            or ckpt_noisy != online_base.noisy
            or ckpt_mlp_depth != online_base.mlp_depth
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
        # Intentionally mutate config to mirror the checkpoint architecture so
        # that subsequent saves/reloads remain self-consistent.
        self.config.grid_channels = ckpt_channels
        self.config.grid_rows = ckpt_rows
        self.config.grid_cols = ckpt_cols
        self.config.dueling = ckpt_dueling
        self.config.noisy = ckpt_noisy
        self.config.mlp_depth = ckpt_mlp_depth

    def _rebuild_networks(
        self,
        grid_size,
        grid_channels,
        hidden_layers,
        grid_rows,
        grid_cols,
        dueling,
        noisy,
        mlp_depth,
    ):
        def _make():
            return QNetwork(
                self.obs_dim,
                self.n_actions,
                grid_size,
                hidden_layers,
                grid_channels,
                grid_rows,
                grid_cols,
                dueling,
                noisy=noisy,
                mlp_depth=mlp_depth,
            ).to(self.device)

        self.online_net, self.target_net, self.inference_net = (
            self._instantiate_networks(_make)
        )
        lr = get_optimizer_lr(self.optimizer)
        self.optimizer = build_optimizer(
            self.online_net, lr, self.device.type, self.use_muon
        )

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
    def _strip_known_prefixes(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        out = dict(state_dict)
        changed = True
        while changed and out:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if all(key.startswith(prefix) for key in out):
                    out = {key[len(prefix) :]: value for key, value in out.items()}
                    changed = True
        return out

    @staticmethod
    def _infer_mlp_depth_from_state_dict(
        state_dict: Optional[Dict[str, torch.Tensor]],
    ) -> int:
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
    def _state_dict_to_cpu(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in state_dict.items()}

    def _load_state_dict_flexible(
        self, module: nn.Module, state_dict: Dict[str, torch.Tensor], name: str
    ) -> None:
        base_module = self._unwrap_module(module)
        normalized_state = self._strip_known_prefixes(state_dict)
        try:
            base_module.load_state_dict(normalized_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load {name} state dict across known key formats: {exc}"
            )
