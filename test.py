"""
Test Harness for VLSI Cell Placement Challenge
==============================================

This script runs the placement optimizer on 10 randomly generated netlists
of various sizes and reports metrics for leaderboard submission.

Usage:
    python test.py

Metrics Reported:
    - Average Overlap: (num cells with overlaps / total num cells)
    - Average Wirelength: (total wirelength / num nets) / sqrt(total area)
      This normalization allows fair comparison across different design sizes.

Note: This test reuses the shared CLI hyperparameter options from placement.py
and evaluates them across the benchmark test cases.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from arg_parse_util import parse_args
from learning_rate_scheduler_util import build_scheduler_kwargs_from_args
from profiler_helper import run_with_optional_profile

# Import from the challenge file
from placement import (
    OUTPUT_DIR,
    calculate_normalized_metrics,
    generate_placement_input,
    get_best_device,
    initialize_cell_positions,
    seed_torch,
    train_placement,
)
from loss_tracking_utils import create_loss_tracking_db, save_loss_history_sqlite


# Test case configurations: (test_id, num_macros, num_std_cells, seed)
TEST_CASES = [
    # Small designs
    (1, 2, 20, 1001),
    (2, 3, 25, 1002),
    (3, 2, 30, 1003),
    # Medium designs
    (4, 3, 50, 1004),
    (5, 4, 75, 1005),
    (6, 5, 100, 1006),
    # Large designs
    (7, 5, 150, 1007),
    (8, 7, 150, 1008),
    (9, 8, 200, 1009),
    # (10, 10, 2000, 1010),
    # Realistic designs
    # (11, 10, 10000, 1011),
    # (12, 10, 100000, 1012),
]


def run_placement_test(
    test_id,
    num_macros,
    num_std_cells,
    loss_tracking_db_path,
    training_config,
    seed=None,
):
    """Run placement optimization on a single test case.

    Args:
        test_id: Test case identifier
        num_macros: Number of macro cells
        num_std_cells: Number of standard cells
        training_config: Hyperparameters for train_placement
        seed: Random seed for reproducibility

    Returns:
        Dictionary with test results and metrics
    """
    if seed:
        # Set seed for reproducibility
        seed_torch(seed)

    device = get_best_device()

    # Generate netlist
    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros,
        num_std_cells,
        device=device,
    )

    # Initialize positions with random spread
    initialize_cell_positions(cell_features)

    # Run optimization with the selected hyperparameters
    start_time = time.time()
    result = train_placement(
        cell_features,
        pin_features,
        edge_list,
        num_epochs=training_config["num_epochs"],
        lr=training_config["lr"],
        lambda_wirelength=training_config["lambda_wirelength"],
        lambda_overlap=training_config["lambda_overlap"],
        scheduler_name=training_config["scheduler_name"],
        scheduler_kwargs=training_config["scheduler_kwargs"],
        track_loss_history=training_config["track_loss_history"],
        verbose=False,  # Suppress per-epoch output
        run_metadata={
            "runner": "test.py",
            "test_id": test_id,
            "seed": seed,
            "num_macros": num_macros,
            "num_std_cells": num_std_cells,
        },
    )
    elapsed_time = time.time() - start_time
    loss_history_path = None
    if training_config["track_loss_history"]:
        loss_history_path = save_loss_history_sqlite(
            result["loss_history"],
            loss_tracking_db_path,
        )

    # Calculate final metrics using shared implementation
    final_cell_features = result["final_cell_features"]
    metrics = calculate_normalized_metrics(final_cell_features, pin_features, edge_list)

    return {
        "test_id": test_id,
        "num_macros": num_macros,
        "num_std_cells": num_std_cells,
        "total_cells": metrics["total_cells"],
        "num_nets": metrics["num_nets"],
        "seed": seed,
        "device": str(device),
        "elapsed_time": elapsed_time,
        "loss_history_path": loss_history_path,
        # Final metrics
        "num_cells_with_overlaps": metrics["num_cells_with_overlaps"],
        "overlap_ratio": metrics["overlap_ratio"],
        "normalized_wl": metrics["normalized_wl"],
    }
def run_placement_test_case_with_config(test_case, loss_tracking_db_path, training_config):
    """Unpack a test-case tuple for multiprocessing execution with config."""
    test_id, num_macros, num_std_cells, seed = test_case
    return run_placement_test(
        test_id,
        num_macros,
        num_std_cells,
        loss_tracking_db_path,
        training_config,
        seed,
    )


def run_all_tests(args):
    """Run all test cases and compute aggregate metrics.

    Returns:
        Dictionary with all test results and aggregate statistics
    """
    training_config = {
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "lambda_wirelength": args.lambda_wirelength,
        "lambda_overlap": args.lambda_overlap,
        "scheduler_name": args.scheduler,
        "scheduler_kwargs": build_scheduler_kwargs_from_args(args),
        "track_loss_history": args.track_loss_history,
    }

    print("=" * 70)
    print("PLACEMENT CHALLENGE TEST SUITE")
    print("=" * 70)
    print(f"\nRunning {len(TEST_CASES)} test cases with various netlist sizes...")
    print("Using hyperparameters:")
    print(f"  num_epochs: {training_config['num_epochs']}")
    print(f"  lr: {training_config['lr']}")
    print(f"  lambda_wirelength: {training_config['lambda_wirelength']}")
    print(f"  lambda_overlap: {training_config['lambda_overlap']}")
    print(f"  scheduler: {training_config['scheduler_name']}")
    print(f"  scheduler_kwargs: {training_config['scheduler_kwargs']}")
    print(f"  track_loss_history: {training_config['track_loss_history']}")
    print()

    loss_tracking_db_path = None
    if args.track_loss_history:
        loss_tracking_db_path = create_loss_tracking_db(OUTPUT_DIR)
        print(f"Writing loss history to: {loss_tracking_db_path}")
        print()
    else:
        print("Loss history tracking disabled.")
        print()

    max_workers = 4

    for idx, (test_id, num_macros, num_std_cells, seed) in enumerate(TEST_CASES, 1):
        size_category = (
            "Small" if num_std_cells <= 30
            else "Medium" if num_std_cells <= 100
            else "Large"
        )

        print(f"Test {idx}/{len(TEST_CASES)}: {size_category} ({num_macros} macros, {num_std_cells} std cells)")
        print(f"  Seed: {seed}")
    print(f"Running up to {max_workers} tests concurrently")
    print()

    wall_start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_test_case = {
            executor.submit(
                run_placement_test_case_with_config,
                test_case,
                loss_tracking_db_path,
                training_config,
            ): test_case
            for test_case in TEST_CASES
        }

        completed_results = {}
        for future in as_completed(future_to_test_case):
            result = future.result()
            completed_results[result["test_id"]] = result

            status = "✓ PASS" if result["num_cells_with_overlaps"] == 0 else "✗ FAIL"
            print(f"Completed test {result['test_id']}:")
            print(
                f"  Device: {result['device']}"
            )
            print(
                f"  Overlap Ratio: {result['overlap_ratio']:.4f} "
                f"({result['num_cells_with_overlaps']}/{result['total_cells']} cells)"
            )
            print(f"  Normalized WL: {result['normalized_wl']:.4f}")
            print(f"  Time: {result['elapsed_time']:.2f}s")
            if result["loss_history_path"] is not None:
                print(f"  History: {result['loss_history_path']}")
            print(f"  Status: {status}")
            print()

    all_results = [
        completed_results[test_id]
        for test_id, _, _, _ in TEST_CASES
    ]

    # Compute aggregate statistics
    avg_overlap_ratio = sum(r["overlap_ratio"] for r in all_results) / len(all_results)
    avg_normalized_wl = sum(r["normalized_wl"] for r in all_results) / len(all_results)
    total_time = time.time() - wall_start_time

    # Print aggregate results
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Average Overlap: {avg_overlap_ratio:.4f}")
    print(f"Average Wirelength: {avg_normalized_wl:.4f}")
    print(f"Total Runtime: {total_time:.2f}s")
    print()

    return {
        "avg_overlap": avg_overlap_ratio,
        "avg_wirelength": avg_normalized_wl,
        "total_time": total_time,
    }


def main(args):
    """Main entry point for the test suite."""
    run_all_tests(args)


if __name__ == "__main__":
    args = parse_args()
    run_with_optional_profile(lambda: main(args), args, OUTPUT_DIR)
