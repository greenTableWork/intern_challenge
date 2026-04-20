import argparse


def parse_args():
    """Parse command line arguments for optional profiling."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile and dump results to the profile directory.",
    )
    parser.add_argument(
        "--profile-tag",
        default="",
        help="Optional tag to include in the profile output filename.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1000,
        help="Number of optimization epochs for a regular training run.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--lambda-wirelength",
        type=float,
        default=3.0,
        help="Weight applied to the wirelength term.",
    )
    parser.add_argument(
        "--lambda-overlap",
        type=float,
        default=1.0,
        help="Weight applied to the overlap term.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["plateau", "cosine", "none"],
        default="plateau",
        help="Learning-rate scheduler to use during training.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=50,
        help="Patience for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="Decay factor for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--scheduler-eta-min",
        type=float,
        default=1e-4,
        help="Minimum learning rate for cosine annealing.",
    )
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="Run Optuna hyperparameter search instead of a single training run.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=25,
        help="Number of Optuna trials to execute.",
    )
    parser.add_argument(
        "--optuna-epochs",
        type=int,
        default=400,
        help="Number of epochs per Optuna trial.",
    )
    parser.add_argument(
        "--optuna-study-name",
        default="placement_hparam_search",
        help="Study name used by Optuna.",
    )
    parser.add_argument(
        "--optuna-storage",
        default="",
        help="Optional Optuna storage URL, for example sqlite:///optuna.db.",
    )
    parser.add_argument(
        "--track-loss-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable loss-history collection and persistence.",
    )
    return parser.parse_args()
