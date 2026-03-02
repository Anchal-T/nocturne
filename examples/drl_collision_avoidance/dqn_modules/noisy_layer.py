import math
import torch
import torch.nn.functional as F
from torch import nn

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        std_init: float = 0.5,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.use_bias = bias

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features, **factory_kwargs))

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_sigma = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self.register_buffer('bias_epsilon', torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
            self.register_buffer('bias_epsilon', None)

        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        if self.use_bias:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        epsilon_in = epsilon_in.to(self.weight_mu.device)
        epsilon_out = epsilon_out.to(self.weight_mu.device)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        if self.use_bias:
            self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = (
                self.bias_mu + self.bias_sigma * self.bias_epsilon
                if self.use_bias
                else None
            )
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.use_bias else None
        return F.linear(x, weight, bias)

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
