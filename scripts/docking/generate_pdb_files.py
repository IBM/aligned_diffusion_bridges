import os
import copy
import argparse
import numpy as np

from Bio.PDB import PDBIO

from proteins.docking.utils import parse_pdb


def update_structure(pdb_file, ca_pred_coords):

    structure = parse_pdb(pdb_file=pdb_file, backend="biopython")
    struct_copy = copy.deepcopy(structure)

    residues = [residue for chain in struct_copy for residue in chain if 'CA' in residue]
    ca_coords = np.asarray([residue['CA'] for residue in residues if 'CA' in residues])
    assert ca_coords.shape[0] == ca_pred_coords.shape[0], "Number of residues must be same as c_alpha predictions"

    for idx, residue in enumerate(residues):
        ca_old = ca_coords[idx]
        ca_new = ca_pred_coords[idx]

        diff = ca_new - ca_old

        for atom in residue:
            old_coord = atom.get_coord()
            new_coord = old_coord + diff
            atom.set_coord(new_coord)

    return struct_copy


def load_residues_from_pdb(struct):
    valid_residues = {}
    count = 0

    for chain_idx, chain in enumerate(struct):
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                # Add water molecule to invalid residues
                continue

            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())

            if c_alpha != None and n != None and c != None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                valid_residues[
                    (chain_idx,) + residue.get_id() + (residue.get_resname(),)
                ] = residue
                count += 1

    residue_ids = valid_residues.keys()
    residue_ids = sorted(residue_ids, key=lambda x: x[:-1], reverse=False)
    
    updated_residues = {residue_id: valid_residues[residue_id] for residue_id in residue_ids}
    return updated_residues        


def generate_prediction_file(pdb_file, ca_coords_0, ca_coords_T, pdb_out_file):
    io = PDBIO()

    struct = parse_pdb(pdb_file=pdb_file, backend="biopython")
    struct_copy = copy.deepcopy(struct)

    residues = load_residues_from_pdb(struct=struct_copy)
    assert len(residues) == len(ca_coords_0), f"{len(residues)}, {len(ca_coords_0)}"
    
    for idx, (residue_id, residue) in enumerate(residues.items()):

        diff = ca_coords_T[idx] - ca_coords_0[idx]

        for atom in residue:
            old_coord = atom.get_coord()
            new_coord = old_coord + diff
            atom.set_coord(new_coord)

    io.set_structure(struct_copy)
    print(f"Saving file to {pdb_out_file}")
    print()
    io.save(pdb_out_file)


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir")
    parser.add_argument("--inference_dir", help="Inference data directory")
    parser.add_argument("--inference_out_dir", help="Inference out directory")
    parser.add_argument("--run_name", help="Run name")

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    input_dir = f"{args.inference_dir}/db5/test_inputs/" 
    traj_dir = f"{args.traj_dir}/db5/{args.run_name}/trajectories/"
    out_dir =  os.path.join(args.inference_out_dir, "db5", "sbalign_results")

    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(traj_dir):
        traj_file_loaded = np.load(f"{traj_dir}/{filename}")

        ca_coords_0 = traj_file_loaded[0]
        ca_coords_T = traj_file_loaded[-1]

        complex_name = filename.split(".")[0]
        pdb_file = os.path.join(input_dir, complex_name + '_l_u_rigid' + '.pdb')
        pdb_out_file = f"{out_dir}/{complex_name}_l_b_SBALIGN.pdb"

        print(f"Performing inference on {pdb_file}")
        try:
            generate_prediction_file(pdb_file=pdb_file, ca_coords_0=ca_coords_0, 
                                ca_coords_T=ca_coords_T, pdb_out_file=pdb_out_file)
        except Exception as e:
            print(e)
            continue

if __name__ == "__main__":
    main()
