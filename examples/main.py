import numpy as np
import pandas as pd
import time
from pathlib import Path
import ripser
import coperto

BASE_DIR = Path(__file__).parent
SEED = 42
N_RUNS = 3
CLIQUE_MAX_SIZE = 3

# File-based datasets
FILE_DATASETS = {
    "celegans": "datasets/celegans.txt",
    "vicsek": "datasets/vicsek_300_of_300.txt",
    "klein_400": "datasets/klein_400.txt",
    "klein_900": "datasets/klein_900.txt",
    "dragon_1k": "datasets/dragon_1000.txt",
    "dragon_2k": "datasets/dragon_2000.txt",
    "HIV1": "datasets/hiv1.txt",
    "o3_1024": "datasets/o3_1024.txt",
    "o3_2048": "datasets/o3_2048.txt",
    "pbmc3k": "datasets/pbmc3k_pca50.txt",
}


def load_dataset(name):
    """Load a dataset by name (file-based)."""
    if name in FILE_DATASETS:
        return np.loadtxt(BASE_DIR / FILE_DATASETS[name])
    else:
        raise ValueError(f"Unknown dataset: {name}")


def measure_time(func, n_runs=3):
    """Helper to measure average execution time."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return np.mean(times)


def run_benchmarks():
    print(
        f"\n=== Running Benchmarks (Runs: {N_RUNS}, Max Clique: {CLIQUE_MAX_SIZE}) ==="
    )
    results = []

    dataset_names = list(FILE_DATASETS.keys())

    for ds_name in dataset_names:
        print(f"Processing {ds_name}...", end=" ", flush=True)
        data = load_dataset(ds_name)
        n_points, n_dims = data.shape

        # --- 1. Calculate Sizes ---
        # Note: Depending on library implementation, these might be fast or slow.
        vr_size = coperto.filtration_size(data, max_size=CLIQUE_MAX_SIZE)
        tower_size = coperto.tower_size(data, max_size=CLIQUE_MAX_SIZE)

        # --- 2. Calculate Times ---
        # Ripser (maxdim=1)
        ripser_time = measure_time(lambda: ripser.ripser(data, maxdim=1), n_runs=N_RUNS)

        # TDA (max_dim=1)
        tda_time = measure_time(
            lambda: coperto.persistent_homology(data, max_dim=1), n_runs=N_RUNS
        )

        print(f"Done. (Ripser: {ripser_time:.3f}s, TDA: {tda_time:.3f}s)")

        # --- 3. Aggregate Results ---
        results.append(
            {
                "dataset": ds_name,
                "n_points": n_points,
                "n_dims": n_dims,
                "vr_size": vr_size,
                "tda_size": tower_size,
                "ripser_time": ripser_time,
                "tda_time": tda_time,
            }
        )

    return pd.DataFrame(results)


def main():
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # Run everything in one loop
    df = run_benchmarks()

    # Save to single CSV
    output_path = results_dir / "main_benchmark.csv"
    df.to_csv(output_path, index=False)

    print("\n=== Final Results ===")
    print(df.to_string())
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
