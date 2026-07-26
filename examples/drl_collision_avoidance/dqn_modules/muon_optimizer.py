import torch
from collections import defaultdict


# Newton-Schulz iteration coefficients for computing the zero-power
# (orthogonal projection) of a matrix. These are the optimal 5th-order
# polynomial coefficients from the Muon optimizer paper.
_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Compute the zero-power (orthogonal projection) of G via Newton-Schulz.

    Works on 2D (M, N) or batched 3D (B, M, N) inputs. For 3D inputs each
    slice in the batch is processed independently via batched matmul.
    """
    assert G.ndim >= 2
    a, b, c = _NS_COEFFICIENTS
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class SingleDeviceMuon(torch.optim.Optimizer):
    """Official Muon for single-GPU (no distributed).

    Batches the Newton-Schulz iterations across identically-shaped
    parameters so that, e.g., all 256x256 weight matrices are stacked
    into one (B, 256, 256) tensor and processed with a single set of
    batched matmuls instead of B separate kernel launches.
    """

    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]

            # Group params by (rows, cols) so we can batch Newton-Schulz.
            shape_groups: dict[tuple, list] = defaultdict(list)
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                shape_groups[p.shape].append(p)

            for shape, plist in shape_groups.items():
                if len(plist) == 1:
                    # Single param — no batching benefit, run directly.
                    p = plist[0]
                    self._update_single(p, group)
                else:
                    # Stack gradients, run batched Newton-Schulz, unstack.
                    self._update_batched(plist, group)

        return loss

    def _update_single(self, p, group):
        state = self.state[p]
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(p)

        g = p.grad
        momentum_buf = state["momentum_buffer"]
        momentum_buf.lerp_(g, 1 - group["momentum"])
        update = g.lerp_(momentum_buf, group["momentum"])

        if update.ndim == 4:
            update = update.view(len(update), -1)
        update = zeropower_via_newtonschulz5(update)
        update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

        p.mul_(1 - group["lr"] * group["weight_decay"])
        p.add_(update.reshape(p.shape), alpha=-group["lr"])

    def _update_batched(self, plist, group):
        # Initialize momentum buffers.
        for p in plist:
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)

        # Stack gradients and momentum buffers into batched tensors.
        grads = torch.stack([p.grad for p in plist])          # (B, M, N)
        moms = torch.stack([self.state[p]["momentum_buffer"] for p in plist])

        # EMA smoothing (vectorized across the batch).
        moms.lerp_(grads, 1 - group["momentum"])
        updates = grads.lerp_(moms, group["momentum"])

        # Reshape 4D conv weights to 2D for Newton-Schulz.
        orig_ndim = updates.ndim
        if orig_ndim == 4:
            updates = updates.view(updates.size(0), updates.size(1), -1)

        # Batched Newton-Schulz — one set of batched matmuls for all params.
        updates = zeropower_via_newtonschulz5(updates)
        scale = max(1, updates.size(-2) / updates.size(-1)) ** 0.5
        updates *= scale

        # Write momentum buffers back and apply updates.
        for i, p in enumerate(plist):
            self.state[p]["momentum_buffer"].copy_(moms[i])
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(updates[i].reshape(p.shape), alpha=-group["lr"])
