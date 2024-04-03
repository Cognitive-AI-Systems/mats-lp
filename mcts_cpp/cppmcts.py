from typing import Optional, Literal

from pogema_toolbox.algorithm_config import AlgoBase
from pydantic import BaseModel
from pydantic import Extra
import gymnasium

import cppimport.import_hook
from mcts_cpp.config import Config
from mcts_cpp.environment import Environment
from mcts_cpp.mcts import Decentralized_MCTS

from env.create_env import DecMAPFConfig


class MCTSConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['MATS-LP'] = 'MATS-LP'
    num_process: int = 1
    gamma: float = 0.96
    num_expansions: int = 250
    steps_limit: int = 128
    use_move_limits: bool = True
    agents_as_obstacles: bool = False
    render: bool = False
    reward_type: Literal['cost2go'] = 'cost2go'
    obs_radius: int = 5
    random_action_chance: float = 0.6
    ppo_only: bool = False
    use_nn_module: bool = True
    agents_to_plan: int = 3
    path_to_weights: str = 'cost-tracer.onnx'
    num_threads: int = 8
    progressed_reward: float = 0.1
    collision_system: Literal['soft'] = 'soft'
    pb_c_init: float = 4.44


class MCTSInference:
    def __init__(self, cfg: MCTSConfig):
        self.cfg = cfg
        self.mcts = Decentralized_MCTS()
        cppconfig = Config()
        cppconfig.gamma = cfg.gamma
        cppconfig.num_expansions = cfg.num_expansions
        cppconfig.steps_limit = cfg.steps_limit
        cppconfig.use_move_limits = cfg.use_move_limits
        cppconfig.agents_as_obstacles = cfg.agents_as_obstacles
        cppconfig.render = cfg.render
        cppconfig.obs_radius = cfg.obs_radius
        cppconfig.random_action_chance = cfg.random_action_chance
        cppconfig.ppo_only = cfg.ppo_only
        cppconfig.use_nn_module = cfg.use_nn_module
        cppconfig.agents_to_plan = cfg.agents_to_plan
        cppconfig.path_to_weights = cfg.path_to_weights
        cppconfig.num_threads = cfg.num_threads
        cppconfig.progressed_reward = cfg.progressed_reward
        cppconfig.pb_c_init = cfg.pb_c_init

        self.cppconfig = cppconfig

    def act(self, observations):

        if 'global_obstacles' in observations[0]:

            gc = DecMAPFConfig(on_target='restart')
            cpp_env = Environment(self.cfg.obs_radius, self.cfg.collision_system, gc.on_target,
                                  self.cfg.progressed_reward)
            cpp_env.create_grid(len(observations[0]['global_obstacles']), len(observations[0]['global_obstacles'][0]))
            for i in range(len(observations[0]['global_obstacles'])):
                for j in range(len(observations[0]['global_obstacles'][0])):
                    if observations[0]['global_obstacles'][i][j]:
                        cpp_env.add_obstacle(i, j)
            cpp_env.precompute_cost2go()
            for agent_idx in range(len(observations)):
                cpp_env.add_agent(observations[agent_idx]['global_agent_xy'],
                                  observations[agent_idx]['global_lifelong_targets_xy'])
            self.mcts.set_config(self.cppconfig)
            cpp_env.set_seed(1)
            self.mcts.set_env(cpp_env, 5)
        action = self.mcts.act()
        return action

    def reset_states(self):
        pass
