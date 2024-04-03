import gymnasium
import numpy as np

from pogema import pogema_v0

import re
from copy import deepcopy
from pogema import GridConfig
from pogema.wrappers.metrics import LifeLongAverageThroughputMetric
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema_toolbox.create_env import Environment

from pogema.generator import generate_new_target
from typing import Literal

from pogema_toolbox.registry import ToolboxRegistry

from env.warehouse_wfi import WarehouseWFI


class DecMAPFConfig(GridConfig):
    collision_system: Literal['soft'] = 'soft'
    on_target: Literal['restart'] = 'restart'
    observation_type: Literal['POMAPF'] = 'POMAPF'
    auto_reset: Literal[False] = False

    num_agents: int = 64
    obs_radius: int = 5
    max_episode_steps: int = 512
    with_animation: bool = False


class ProvideGlobalObstacles(gymnasium.Wrapper):
    def get_global_obstacles(self):
        return self.grid.get_obstacles().astype(int).tolist()

    def get_global_agents_xy(self):
        return self.grid.get_agents_xy()

    def get_global_targets_xy(self):
        return self.grid.get_targets_xy()

    def get_lifelong_global_targets_xy(self):
        all_goals = []
        cur_goals = self.grid.get_targets_xy()
        generators = deepcopy(self.random_generators)
        for agent_idx in range(self.grid_config.num_agents):
            distance = 0
            cur_goal = cur_goals[agent_idx]
            goals = [cur_goal]
            while distance < self.grid_config.max_episode_steps + 100:
                new_goal = generate_new_target(generators[agent_idx],
                                               self.grid.point_to_component,
                                               self.grid.component_to_points,
                                               self.grid.positions_xy[agent_idx])
                distance += abs(cur_goal[0] - new_goal[0]) + abs(cur_goal[1] - new_goal[1])
                cur_goal = new_goal
                goals.append(cur_goal)
            all_goals.append(goals)
        return all_goals


class ProvideMapWrapper(gymnasium.Wrapper):
    def reset(self, **kwargs):
        observations, infos = self.env.reset(seed=self.env.grid_config.seed)
        global_obstacles = self.get_global_obstacles()
        global_agents_xy = self.get_global_agents_xy()
        global_targets_xy = self.get_global_targets_xy()
        global_lifelong_targets_xy = self.get_lifelong_global_targets_xy()
        for idx, obs in enumerate(observations):
            obs['global_obstacles'] = global_obstacles
            obs['global_agent_xy'] = global_agents_xy[idx]
            obs['global_target_xy'] = global_targets_xy[idx]
            obs['global_lifelong_targets_xy'] = global_lifelong_targets_xy[idx]
        return observations, infos


def create_env_base(env_cfg: Environment):
    config = env_cfg.grid_config
    if config.map_name == 'wfi_warehouse':
        env = WarehouseWFI(grid_config=config)
        env = ProvideGlobalObstacles(env)
        env = ProvideMapWrapper(env)
        env = MultiTimeLimit(env, config.max_episode_steps)
        env = LifeLongAverageThroughputMetric(env)
    else:
        env = pogema_v0(grid_config=config)
        env = ProvideGlobalObstacles(env)
        env = ProvideMapWrapper(env)
        env = MultiMapWrapper(env)
    return env


class MultiMapWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._configs = []
        self._rnd = np.random.default_rng(self.grid_config.seed)
        pattern = self.grid_config.map_name

        if pattern:
            maps = ToolboxRegistry.get_maps()
            for map_name in sorted(maps):
                if re.match(pattern, map_name):
                    cfg = deepcopy(self.grid_config)
                    cfg.map = maps[map_name]
                    cfg.map_name = map_name
                    cfg = GridConfig(**cfg.dict())
                    self._configs.append(cfg)
            if not self._configs:
                raise KeyError(f"No map matching: {pattern}")

    def reset(self, seed=None, **kwargs):
        self._rnd = np.random.default_rng(seed)
        if self._configs is not None and len(self._configs) >= 1:
            map_idx = self._rnd.integers(0, len(self._configs))
            cfg = deepcopy(self._configs[map_idx])
            self.env.unwrapped.grid_config = cfg
            self.env.unwrapped.grid_config.seed = seed
        return self.env.reset(seed=seed, **kwargs)
