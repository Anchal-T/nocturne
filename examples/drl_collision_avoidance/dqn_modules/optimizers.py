from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from .muon_optimizer import SingleDeviceMuon


class HybridOptimizer:
    """Wraps Muon (for hidden Linear weights) + AdamW (for everything else).

    Provides the same zero_grad / step / state_dict interface as a regular
    PyTorch optimizer so callers don't need to care about the split.
    """

    def __init__(self, muon: SingleDeviceMuon, adamw: optim.AdamW):
        self.muon = muon
        self.adamw = adamw

    def zero_grad(self, set_to_none: bool = False):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def _make_adam(params: List[nn.Parameter], lr: float, device_type: str) -> optim.Adam:
    kwargs = {"lr": float(lr)}
    if device_type == "cuda":
        kwargs["fused"] = True
    try:
        return optim.Adam(params, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return optim.Adam(params, **kwargs)


def _make_adamw(params: List[nn.Parameter], lr: float, device_type: str) -> optim.AdamW:
    kwargs = {"lr": lr * 0.5, "weight_decay": 0.01}
    if device_type == "cuda":
        kwargs["fused"] = True
    try:
        return optim.AdamW(params, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return optim.AdamW(params, **kwargs)


def build_optimizer(
    network: nn.Module,
    lr: float,
    device_type: str,
    use_muon: bool,
) -> object:
    """Build the training optimizer for the given network.

    When use_muon is True, hidden 2D Linear weights get Muon and everything
    else gets AdamW. Falls back to plain Adam if Muon has no eligible params.
    """
    if not use_muon:
        return _make_adam(list(network.parameters()), lr, device_type)

    muon_params = []
    adamw_params = []
    # Muon convention: 2D weight matrices from hidden layers get Muon,
    # but final output heads and non-weight tensors get AdamW.
    output_head_markers = ("advantage_head.2", "value_head.2", "head.2")
    for name, param in network.named_parameters():
        is_hidden_weight = param.ndim == 2 and "weight" in name
        is_output_head = any(m in name for m in output_head_markers)
        if is_hidden_weight and not is_output_head:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    if not muon_params:
        return _make_adam(list(network.parameters()), lr, device_type)

    muon_opt = SingleDeviceMuon(
        muon_params,
        lr=lr,
        weight_decay=0.1,
        momentum=0.95,
    )
    adamw_opt = _make_adamw(adamw_params, lr, device_type)
    return HybridOptimizer(muon_opt, adamw_opt)


def get_optimizer_lr(optimizer) -> float:
    if isinstance(optimizer, HybridOptimizer):
        return float(optimizer.muon.param_groups[0]["lr"])
    return float(optimizer.param_groups[0]["lr"])
