import torch


def zeropower_via_newtonschulz5(G, steps: int = 5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
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
    """Official Muon for single-GPU (no distributed)."""
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
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                g = p.grad
                momentum_buf = state["momentum_buffer"]
                momentum_buf.lerp_(g, 1 - group["momentum"])
                update = g.lerp_(momentum_buf, group["momentum"]) if True else momentum_buf   # nesterov=True

                if update.ndim == 4:  # conv case (rare here)
                    update = update.view(len(update), -1)
                update = zeropower_via_newtonschulz5(update)
                update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
        return loss