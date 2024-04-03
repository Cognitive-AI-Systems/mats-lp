import argparse

import yaml
from pogema_toolbox.create_env import Environment
from pogema_toolbox.registry import ToolboxRegistry

from env.create_env import create_env_base, DecMAPFConfig
from mcts_cpp.cppmcts import MCTSConfig, MCTSInference


def main():
    parser = argparse.ArgumentParser(description='MCTS Inference Script')
    parser.add_argument('--animation', action='store_false', help='Enable animation (default: %(default)s)')
    parser.add_argument('--num_agents', type=int, default=8, help='Number of agents (default: %(default)d)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: %(default)d)')
    parser.add_argument('--map_name', type=str, default='pico_s00_od20_na32', help='Map name (default: %(default)s)')
    parser.add_argument('--max_episode_steps', type=int, default=64,
                        help='Maximum episode steps (default: %(default)d)')
    parser.add_argument('--num_expansions', type=int, default=250,
                        help='Number of MCTS expansions (default: %(default)d)')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads (default: %(default)d)')
    parser.add_argument('--pb_c_init', type=float, default=4.44,
                        help='UCT exploration initial value (default: %(default)f)')
    parser.add_argument('--show_map_names', action='store_true', help='Shows names of all available maps')

    args = parser.parse_args()

    with open("env/mazes-maps.yaml", 'r') as f:
        maps_to_register = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps_to_register)
    with open("env/random-pico.yaml", 'r') as f:
        maps_to_register = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps_to_register)

    if args.show_map_names:
        for map_ in ToolboxRegistry.get_maps():
            print(map_)
        print('wfi_warehouse')
        return

    env_cfg = Environment(grid_config=DecMAPFConfig(
        with_animation=args.animation,
        num_agents=args.num_agents,
        seed=args.seed,
        map_name=args.map_name,
        max_episode_steps=args.max_episode_steps
    ))

    algo = MCTSInference(MCTSConfig(
        num_expansions=args.num_expansions,
        num_threads=args.num_threads,
        pb_c_init=args.pb_c_init))
    env = create_env_base(env_cfg)

    obs, _ = env.reset(seed=env.grid_config.seed)
    while True:
        obs, rew, dones, tr, infos = env.step(algo.act(obs))
        if all(dones) or all(tr):
            break

    print(infos[0]['metrics'])


if __name__ == '__main__':
    main()
