import argparse
import torch
import json
import os
import numpy as np

from biopandas.pdb import PandasPdb

from sbalign.utils.ops import to_numpy
from sbalign.utils.definitions import DEVICE


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir")
    parser.add_argument("--dataset")
    parser.add_argument("--inference_dir")

    parser.add_argument("--load_dir")
    parser.add_argument("--rigid", action='store_true')
    
    args = parser.parse_args()
    return args


def regen_ids_from_zero(ppdb, field):
    field_numbers = ppdb._df['ATOM'][field].to_numpy().copy()
    cur_id = 1
    cur_field_num = field_numbers[0]
    for v in range(field_numbers.shape[0]):
        if field_numbers[v] == cur_field_num:
            field_numbers[v] = cur_id
        else:
            cur_field_num = field_numbers[v]
            cur_id += 1
            field_numbers[v] = cur_id
    ppdb._df['ATOM'][field] = field_numbers
    return ppdb


def random_transf_pdb(ppdb, pdb_out_file, R, tr, unchanged: bool = False):
    if not unchanged:
        atom_loc = ppdb._df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
        assert np.mean(atom_loc, axis=0).shape[0] == 3
        # Randomly rotate and translate the ligand.
        atom_mean = np.mean(atom_loc, axis=0, keepdims=True)
        new_atom_loc = (R @ (atom_loc - atom_mean).T).T + atom_mean + tr

        ppdb._df['ATOM']['x_coord'] = new_atom_loc[:, 0]
        ppdb._df['ATOM']['y_coord'] = new_atom_loc[:, 1]
        ppdb._df['ATOM']['z_coord'] = new_atom_loc[:, 2]

        assert np.linalg.norm(new_atom_loc - atom_loc) > 0.1
        assert np.linalg.norm(ppdb._df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy() - atom_loc) > 0.1

    ppdb = regen_ids_from_zero(ppdb, 'residue_number')
    ppdb = regen_ids_from_zero(ppdb, 'atom_number')

    print(f"Saving pdb to file {pdb_out_file}")

    ppdb.to_pdb(path=pdb_out_file,
                records=['ATOM'],
                gz=False,
                append_newline=True)


def generate_test_files(args):

    rot_tr_file = f"{args.load_dir}/R_tr.pt"
    rot_tr_dict = torch.load(rot_tr_file, map_location='cpu')

    R, tr = rot_tr_dict['R'], rot_tr_dict['tr']
    R = to_numpy(R)
    tr = to_numpy(tr)

    complex_dir = f"{args.data_dir}/raw/{args.dataset}/complexes/"
    with open(f"{args.data_dir}/raw/{args.dataset}/splits.json", "r") as f:
        splits = json.load(f)
    test_complexes = splits['test']
    
    inference_dir = f"{args.inference_dir}/{args.dataset}/test_inputs/"
    inference_gt_dir = f"{args.inference_dir}/{args.dataset}/test_complexes/"

    os.makedirs(inference_dir, exist_ok=True)
    os.makedirs(inference_gt_dir, exist_ok=True)

    for complex in test_complexes:
        print(f"Complex: {complex}", flush=True)

        if not args.rigid:
            lig_u_file = f"{complex_dir}/{complex}_l_u.pdb"
            lig_u_out_file = f"{inference_dir}/{complex}_l_u_flexible.pdb"

            lig_u_ppdb = PandasPdb().read_pdb(lig_u_file)
            random_transf_pdb(lig_u_ppdb, pdb_out_file=lig_u_out_file, R=R, tr=tr,
                            unchanged=False)
        
        else:
            lig_b_file = f"{complex_dir}/{complex}_l_b.pdb"
            lig_b_out_file = f"{inference_dir}/{complex}_l_u_rigid.pdb"
            lig_b_ppdb = PandasPdb().read_pdb(lig_b_file)
            random_transf_pdb(lig_b_ppdb, pdb_out_file=lig_b_out_file, R=R, tr=tr,
                            unchanged=False)

        lig_b_file = f"{complex_dir}/{complex}_l_b.pdb"
        lig_b_out_file = f"{inference_gt_dir}/{complex}_l_b_COMPLEX.pdb"
        lig_b_ppdb = PandasPdb().read_pdb(lig_b_file)
        random_transf_pdb(lig_b_ppdb, pdb_out_file=lig_b_out_file, R=R, tr=tr,
                         unchanged=True)

        rec_u_file = f"{complex_dir}/{complex}_r_u.pdb"
        rec_u_out_file = f"{inference_gt_dir}/{complex}_r_u_COMPLEX.pdb"
        rec_u_ppdb = PandasPdb().read_pdb(rec_u_file)
        random_transf_pdb(rec_u_ppdb, pdb_out_file=rec_u_out_file, R=R, tr=tr,
                        unchanged=True)   

        rec_b_file = f"{complex_dir}/{complex}_r_b.pdb"
        rec_b_out_file = f"{inference_gt_dir}/{complex}_r_b_COMPLEX.pdb"
        rec_b_ppdb = PandasPdb().read_pdb(rec_b_file)
        random_transf_pdb(rec_b_ppdb, pdb_out_file=rec_b_out_file, R=R, tr=tr,
                         unchanged=True)
        print(flush=True)


if __name__ == "__main__":
    args = parse_args()

    print("Args: ", args)

    generate_test_files(args)
