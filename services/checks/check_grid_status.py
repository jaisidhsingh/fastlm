"""
Check the state of the OpenThesis training grid on Hugging Face
and visualize progress directly in the terminal.

Grid:
    Architecture: 5 values
    N:            20M, 50M, 150M, 300M
    GBS:          16, 32, 64, 128, 256
    LR:           0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008
    D:            0.5B, 1.0B, 3.0B, 7.5B, 15.0B

Expected artifact:
    OpenThesis_{arch_id}/
        N/
            gbs_{GBS}/
                lr_{LR}/
                    ckpt_decayed_to_{D}.pt

Usage:
    pip install huggingface_hub

    # If already logged into HF:
    python check_grid.py

    # Or specify the HF namespace:
    python check_grid.py --namespace your-hf-username

The script assumes the repositories are dataset repositories.
Change REPO_TYPE below if they are model repositories.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from huggingface_hub import HfApi


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ARCH_IDS = [
    "attn",
    "gdn-attn_1-3",
    "gdn-attn_1-1",
    "gdn-attn_3-1",
    "gdn",
]

MODEL_SIZES = [
    "20M",
    "50M",
    "150M",
    "300M",
]

GBS_VALUES = [
    16,
    32,
    64,
    128,
    256,
]

LR_VALUES = [
    0.00025,
    0.0005,
    0.001,
    0.002,
    0.004,
    0.008,
]

TOKEN_BUDGETS = [
    "0.5B",
    "1.0B",
    "3.0B",
    "7.5B",
    "15.0B",
]

REPO_PREFIX = "OpenThesis_"

# Your repos appear to be dataset repos.
REPO_TYPE = "dataset"

OUTPUT_MISSING = "missing_grid_points.csv"


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def format_lr(lr: float) -> str:
    """
    Match the LR formatting used in the HF path.

    Examples:
        0.00025 -> "0.00025"
        0.001   -> "0.001"
    """
    return str(lr).replace(".", "p") 


def checkpoint_path(
    N: str,
    gbs: int,
    lr: float,
    D: str,
) -> str:
    return (
        f"{N}/"
        f"gbs_{gbs}/"
        f"lr_{format_lr(lr)}/"
        f"ckpt_decayed_to_{D.replace('.', 'p')}.pt"
    )


# ---------------------------------------------------------------------
# HF inspection
# ---------------------------------------------------------------------

def get_repo_files(
    api: HfApi,
    repo_id: str,
) -> set[str]:
    """
    Retrieve all file paths from one HF repository.

    The entire file listing is fetched once per architecture.
    """
    try:
        files = api.list_repo_files(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )
        return set(files)

    except Exception as exc:
        print(f"[WARNING] Could not access {repo_id}: {exc}")
        return set()


def check_grid(
    api: HfApi,
    namespace: str,
):
    """
    Return:
        completed: set of (arch, N, GBS, LR, D)
        missing:   list of missing tuples
        repo_files: dict[arch, set[path]]
    """

    repo_files = {}

    # -------------------------------------------------------------
    # Download/list each repository exactly once.
    # -------------------------------------------------------------

    for arch in ARCH_IDS:
        repo_id = f"{namespace}/{REPO_PREFIX}{arch}"

        print(f"Checking {repo_id} ...")

        repo_files[arch] = get_repo_files(
            api,
            repo_id,
        )

        print(
            f"    found {len(repo_files[arch]):,} files"
        )

    # -------------------------------------------------------------
    # Check every point in the Cartesian product.
    # -------------------------------------------------------------

    completed = set()
    missing = []

    all_points = itertools.product(
        ARCH_IDS,
        MODEL_SIZES,
        GBS_VALUES,
        LR_VALUES,
        TOKEN_BUDGETS,
    )

    for arch, N, gbs, lr, D in all_points:

        path = checkpoint_path(
            N=N,
            gbs=gbs,
            lr=lr,
            D=D,
        )

        point = (
            arch,
            N,
            gbs,
            lr,
            D,
        )

        if path in repo_files[arch]:
            completed.add(point)
        else:
            missing.append(point)

    return completed, missing, repo_files


# ---------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------

def write_missing_csv(missing):
    with open(
        OUTPUT_MISSING,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "arch_id",
            "N",
            "GBS",
            "LR",
            "D",
            "path",
        ])

        for arch, N, gbs, lr, D in missing:
            writer.writerow([
                arch,
                N,
                gbs,
                lr,
                D,
                checkpoint_path(
                    N,
                    gbs,
                    lr,
                    D,
                ),
            ])

    print(f"\nMissing grid points written to {OUTPUT_MISSING}")


# ---------------------------------------------------------------------
# Terminal progress visualization
# ---------------------------------------------------------------------

def terminal_progress(completed):
    """
    Print one terminal heatmap per architecture and model size.

    Rows    : GBS
    Columns : LR
    Cell    : five-character D completion mask

        □□□□■  -> only 15.0B is complete
        ■■■□□  -> 0.5B, 1.0B, 3.0B are complete
        ■■■■■  -> all token budgets are complete

    This preserves the full (N, GBS, LR, D) structure while remaining
    readable over SSH on a remote cluster.
    """

    print()
    print("=" * 100)
    print("GRID PROGRESS")
    print("=" * 100)

    for arch in ARCH_IDS:
        print()
        print(f"ARCHITECTURE: {arch}")
        print("-" * 100)

        for N in MODEL_SIZES:
            print()
            print(f"  N = {N}")
            print()

            # Header
            lr_labels = [format_lr(lr) for lr in LR_VALUES]

            print(
                f"  {'GBS':>5} | "
                + " | ".join(f"{lr:>7}" for lr in lr_labels)
            )
            print(
                "  "
                + "-" * (
                    7 + len(LR_VALUES) * 10
                )
            )

            for gbs in GBS_VALUES:
                cells = []

                for lr in LR_VALUES:
                    mask = ""

                    for D in TOKEN_BUDGETS:
                        point = (
                            arch,
                            N,
                            gbs,
                            lr,
                            D,
                        )

                        mask += "■" if point in completed else "□"

                    cells.append(mask)

                print(
                    f"  {gbs:>5} | "
                    + " | ".join(f"{cell:^7}" for cell in cells)
                )

            print()
            print(
                "  D: "
                + "  ".join(
                    f"{i + 1}={D}"
                    for i, D in enumerate(TOKEN_BUDGETS)
                )
            )

    # Summary by architecture.
    print()
    print("=" * 100)
    print("ARCHITECTURE SUMMARY")
    print("=" * 100)

    total_per_arch = (
        len(MODEL_SIZES)
        * len(GBS_VALUES)
        * len(LR_VALUES)
        * len(TOKEN_BUDGETS)
    )

    for arch in ARCH_IDS:
        count = sum(
            1
            for point in completed
            if point[0] == arch
        )

        percentage = (
            100.0 * count / total_per_arch
            if total_per_arch
            else 0.0
        )

        print(
            f"{arch:<16} "
            f"{count:>5}/{total_per_arch:<5} "
            f"({percentage:6.2f}%)"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check OpenThesis HF training grid."
    )

    parser.add_argument(
        "--namespace",
        default="jaisidhsingh",
        help="Hugging Face username or organization containing the repos.",
    )

    args = parser.parse_args()

    api = HfApi()

    total = (
        len(ARCH_IDS)
        * len(MODEL_SIZES)
        * len(GBS_VALUES)
        * len(LR_VALUES)
        * len(TOKEN_BUDGETS)
    )

    print("=" * 70)
    print("OpenThesis Grid Check")
    print("=" * 70)

    print(f"Architectures : {len(ARCH_IDS)}")
    print(f"Model sizes   : {len(MODEL_SIZES)}")
    print(f"GBS values    : {len(GBS_VALUES)}")
    print(f"LR values     : {len(LR_VALUES)}")
    print(f"Token budgets : {len(TOKEN_BUDGETS)}")
    print(f"Total grid    : {total:,}")
    print()

    completed, missing, _ = check_grid(
        api,
        args.namespace,
    )

    percentage = 100.0 * len(completed) / total

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Complete : {len(completed):,} / {total:,}")
    print(f"Missing  : {len(missing):,} / {total:,}")
    print(f"Progress : {percentage:.2f}%")
    print("=" * 70)

    write_missing_csv(missing)

    terminal_progress(completed)


if __name__ == "__main__":
    main()

