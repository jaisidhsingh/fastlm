import os
import sys
from functools import partial

from datasets import Dataset, load_dataset, load_from_disk

DATASET_NAME = "MultiSynt/Nemotron-CC-sample-2"
OUTPUT_DIR = "/data/horse/ws/jasi149i-fastlm"


def main():
    num_proc = int(sys.argv[1])
    print(f"Downloading {DATASET_NAME}...")

    raw_ds = load_dataset(
        DATASET_NAME,
        split="train",
        cache_dir=OUTPUT_DIR + "/hf_cache",
        # num_proc=num_proc,
        streaming=True,
    )

    def custom_generator(iterable_ds):
        yield from iterable_ds

    raw_ds = Dataset.from_generator(
        partial(custom_generator, raw_ds),
        features=raw_ds.features,
    )

    save_path = OUTPUT_DIR + "/data/nemotron-cc-sample-mtsynth/raw_dataset"
    os.makedirs(save_path, exist_ok=True)
    raw_ds.save_to_disk(save_path)
    print(f"Saved to {save_path}")

    new_ds = load_from_disk(save_path)
    print(len(new_ds))


if __name__ == "__main__":
    main()
