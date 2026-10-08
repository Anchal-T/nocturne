"""Parameter counts for the DDQN with and without the four occlusion features.

The grid encoder is shared. The four features enter only the dueling heads, so
the difference is 4,096 parameters for the default noisy dueling network
(2 heads x 2 noisy matrices x 256 x 4).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from examples.drl_collision_avoidance.observation_config import head_parameter_delta


GRID = 3 * 20 * 10
BASE_EXTRA = 4 + 2 + 3  # ego, goal, TTZ
OCC_EXTRA = 4


def main():
    delta = head_parameter_delta(OCC_EXTRA, head_dim=256, dueling=True, noisy=True)
    print(f'observation without occlusion features: {GRID + BASE_EXTRA}')
    print(f'observation with occlusion features:    {GRID + BASE_EXTRA + OCC_EXTRA}')
    print(f'parameter difference (analytical):      {delta}')

    import torch
    from examples.drl_collision_avoidance.dqn_modules.q_network import QNetwork

    def count(obs_dim):
        net = QNetwork(
            obs_dim=obs_dim,
            n_actions=15,
            grid_size=GRID,
            grid_channels=3,
            grid_rows=20,
            grid_cols=10,
            hidden_layers=[256, 256, 256, 256],
            dueling=True,
            noisy=True,
            mlp_depth=16,
        )
        return sum(param.numel() for param in net.parameters())

    without = count(GRID + BASE_EXTRA)
    with_occ = count(GRID + BASE_EXTRA + OCC_EXTRA)
    print(f'parameters without occlusion features:  {without}')
    print(f'parameters with occlusion features:     {with_occ}')
    print(f'parameter difference (constructed):     {with_occ - without}')
    if with_occ - without != delta:
        raise SystemExit('constructed parameter delta does not match the analytical count')


if __name__ == '__main__':
    main()
