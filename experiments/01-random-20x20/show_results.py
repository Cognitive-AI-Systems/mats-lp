from pathlib import Path
from pogema_toolbox.views.view_utils import load_from_folder, check_seeds

from pogema_toolbox.evaluator import run_views


def format_results(results):
    return results


def main():
    current_dir_name = Path(__file__).parent

    results, evaluation_config = load_from_folder(current_dir_name)
    results = format_results(results)

    run_views(results, evaluation_config, eval_dir=current_dir_name)
    check_seeds(results)


if __name__ == '__main__':
    main()
