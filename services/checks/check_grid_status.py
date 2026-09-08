"""Print incomplete OpenThesis grid points found on Hugging Face.

Each grid point is complete only when both of these files exist:

    OpenThesis_{arch_id}/N/gbs_{GBS}/lr_{LR}/ckpt_decayed_to_{D}.pt
    OpenThesis_{arch_id}/N/gbs_{GBS}/lr_{LR}/metrics_decayed_to_{D}.json

GBS 16 expects token budgets up to 1.0B, while GBS 32 expects budgets up to 3.0B.

Usage:
    python services/checks/check_grid_status.py --gbs 128
    python services/checks/check_grid_status.py --gbs 128 --namespace my-hf-org
"""

from __future__ import annotations

import argparse
import itertools

from huggingface_hub import HfApi


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
REPO_TYPE = "dataset"

type MissingPoint = tuple[str, str, int, float, str, tuple[str, ...]]


def format_lr(lr: float) -> str:
    return str(lr).replace(".", "p")


def artifact_path(
    N: str,
    gbs: int,
    lr: float,
    D: str,
    filename: str,
) -> str:
    return (
        f"{N}/"
        f"gbs_{gbs}/"
        f"lr_{format_lr(lr)}/"
        f"{filename}_decayed_to_{D.replace('.', 'p')}"
    )


def checkpoint_path(N: str, gbs: int, lr: float, D: str) -> str:
    return artifact_path(N, gbs, lr, D, "ckpt") + ".pt"


def metrics_path(N: str, gbs: int, lr: float, D: str) -> str:
    return artifact_path(N, gbs, lr, D, "metrics") + ".json"


def token_budgets_for_gbs(gbs: int) -> list[str]:
    if gbs == 16:
        return TOKEN_BUDGETS[:2]
    if gbs == 32:
        return TOKEN_BUDGETS[:3]
    return TOKEN_BUDGETS


def get_repo_files(api: HfApi, repo_id: str) -> set[str]:
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
    max_gbs: int,
) -> list[MissingPoint]:
    repo_files = {
        arch: get_repo_files(api, f"{namespace}/{REPO_PREFIX}{arch}")
        for arch in ARCH_IDS
    }
    selected_gbs_values = [gbs for gbs in GBS_VALUES if gbs <= max_gbs]
    missing = []

    for arch, N, gbs, lr in itertools.product(
        ARCH_IDS,
        MODEL_SIZES,
        selected_gbs_values,
        LR_VALUES,
    ):
        for D in token_budgets_for_gbs(gbs):
            expected_paths = {
                "checkpoint": checkpoint_path(N, gbs, lr, D),
                "metrics": metrics_path(N, gbs, lr, D),
            }
            missing_artifacts = tuple(
                name
                for name, path in expected_paths.items()
                if path not in repo_files[arch]
            )
            if missing_artifacts:
                missing.append((arch, N, gbs, lr, D, missing_artifacts))

    return missing


def print_missing_points(missing: list[MissingPoint]) -> None:
    for arch in ARCH_IDS:
        architecture_points = [point for point in missing if point[0] == arch]
        if not architecture_points:
            continue

        print(arch)
        for N in MODEL_SIZES:
            for gbs in GBS_VALUES:
                points = [
                    point
                    for point in architecture_points
                    if point[1] == N and point[2] == gbs
                ]
                if not points:
                    continue

                entire_block_missing = (
                    len(points)
                    == len(LR_VALUES) * len(token_budgets_for_gbs(gbs))
                    and all(
                        point[5] == ("checkpoint", "metrics")
                        for point in points
                    )
                )
                if entire_block_missing:
                    print(
                        f"  N={N}, GBS={gbs}: all LR x D points missing "
                        "(checkpoint and metrics)"
                    )
                    continue

                for _, _, _, lr, D, missing_artifacts in points:
                    artifacts = " and ".join(missing_artifacts)
                    print(
                        f"  N={N}, GBS={gbs}, LR={lr}, D={D}: "
                        f"missing {artifacts}"
                    )

        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print incomplete OpenThesis Hugging Face grid points."
    )
    parser.add_argument(
        "--namespace",
        default="jaisidhsingh",
        help="Hugging Face username or organization containing the repos.",
    )
    parser.add_argument(
        "--gbs",
        type=int,
        required=True,
        help="Check configured global batch sizes less than or equal to this value.",
    )

    args = parser.parse_args()
    if args.gbs < min(GBS_VALUES):
        parser.error(f"--gbs must be at least {min(GBS_VALUES)}")

    missing = check_grid(HfApi(), args.namespace, args.gbs)
    print_missing_points(missing)


if __name__ == "__main__":
    main()
