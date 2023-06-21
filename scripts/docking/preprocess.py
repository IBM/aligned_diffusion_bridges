import argparse

from proteins.docking.dataset import RigidProteinDocking


def parse_preprocess_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir")
    parser.add_argument("--dataset", default="db5", help="Dataset to process")
    parser.add_argument("--backend", default="biopandas", type=str)

    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--progress_every", default=500, type=int)

    # Ligand and receptor specific arguments
    parser.add_argument("--resolution", default="c_alpha", choices=["all", "bb", "c_alpha"])

    args = parser.parse_args()
    return args


def main():
    args = parse_preprocess_args()

    dataset = RigidProteinDocking(root=args.data_dir, transform=None,
                                  dataset=args.dataset, split_mode="preprocess",
                                  resolution=args.resolution, 
                                  num_workers=args.num_workers,
                                  backend=args.backend,
                                  progress_every=100)
    dataset.preprocess_complexes()

if __name__ == "__main__":
    main()
