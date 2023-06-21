import argparse

from proteins.conf.dataset import ProteinConfDataset


def parse_preprocess_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir")
    parser.add_argument("--dataset", default="d3pm", help="Dataset to process")

    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--progress_every", default=500, type=int)
    parser.add_argument("--center_conformations", action="store_true")

    parser.add_argument("--resolution", default="c_alpha", choices=["all", "bb", "c_alpha"])

    args = parser.parse_args()
    return args


def main():
    args = parse_preprocess_args()

    dataset = ProteinConfDataset(root=args.data_dir, transform=None,
                                 dataset=args.dataset, split_mode="preprocess",
                                 resolution=args.resolution,
                                 num_workers=args.num_workers,
                                 center_conformations=args.center_conformations,
                                 progress_every=args.progress_every
                                )

    dataset.preprocess_conformation_pairs()

if __name__ == "__main__":
    main()
