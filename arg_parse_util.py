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
    return parser.parse_args()
