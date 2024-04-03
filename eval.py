from pogema import BatchAStarAgent
from pogema_toolbox.create_env import Environment
from pogema_toolbox.evaluator import evaluation

from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results

from pathlib import Path

import yaml

from pogema_toolbox.registry import ToolboxRegistry

from env.create_env import create_env_base
from mcts_cpp.cppmcts import MCTSInference, MCTSConfig

PROJECT_NAME = 'pogema-toolbox'
BASE_PATH = Path('experiments')


def main(disable_wandb=True):
    ToolboxRegistry.register_env('Pogema-v0', create_env_base, Environment)
    ToolboxRegistry.register_algorithm('MATS-LP', MCTSInference, MCTSConfig, )
    ToolboxRegistry.register_algorithm('A*', BatchAStarAgent)

    with open("env/mazes-maps.yaml", 'r') as f:
        maps_to_register = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps_to_register)
    with open("env/random-pico.yaml", 'r') as f:
        maps_to_register = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps_to_register)

    folder_names = [
        '01-random-20x20',
    ]

    for folder in folder_names:
        config_path = BASE_PATH / folder / f"{Path(folder).name}.yaml"
        eval_dir = BASE_PATH / folder

        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)
        if folder == 'eval-fast':
            disable_wandb = True

        initialize_wandb(evaluation_config, eval_dir, disable_wandb, PROJECT_NAME)
        evaluation(evaluation_config, eval_dir=eval_dir)
        save_evaluation_results(eval_dir)


if __name__ == '__main__':
    main()
