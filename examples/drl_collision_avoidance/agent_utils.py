from typing import Any, Dict

from examples.drl_collision_avoidance.dqn_modules.ddqn_agent import DDQNAgent, DDQNAgentConfig


def build_cpu_agent(cfg: Dict[str, Any], obs_dim: int, n_actions: int) -> DDQNAgent:
    """Build a DDQNAgent on CPU for evaluation/visualization."""
    grid_cfg = cfg['occupancy_grid']
    grid_channels = 3
    grid_rows = int(grid_cfg['rows'])
    grid_cols = int(grid_cfg['cols'])
    grid_size = grid_channels * grid_rows * grid_cols
    drl_cfg = cfg['drl']

    agent_cfg = DDQNAgentConfig.from_drl_cfg(
        drl_cfg,
        grid_size=grid_size,
        grid_channels=grid_channels,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        device='cpu',
    )
    return DDQNAgent(obs_dim=obs_dim, n_actions=n_actions, config=agent_cfg)
