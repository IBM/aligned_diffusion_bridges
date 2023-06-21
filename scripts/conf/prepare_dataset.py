import os
import argparse
import traceback

import pandas as pd
import numpy as np

from proteins.conf.utils import (
    get_residue_dict, align_residue_dicts, parse_cif_biopandas,
    gather_ca_coords
)
from proteins.docking.metrics import rigid_transform_kabsch_3D, rmsd


def rmsd_check(args, rmsd_align, rmsd_true):
    if np.abs(rmsd_align - rmsd_true) / rmsd_true > args.rmsd_tol:
        return False
    
    return True


def load_conformation_df(args):
    conf_df = pd.read_excel(f"{args.data_dir}/raw/{args.dataset}/{args.dataset}.xlsx")
    conf_df["Structure_apo"] = conf_df["Structure_apo"].astype(str).str.strip()
    conf_df["Structure_bound"] = conf_df["Structure_bound"].astype(str).str.strip()
    conf_df["RMSD_overall"] =conf_df["RMSD_overall"].astype(float)
    return conf_df



def load_cif_and_residues(conformation_dir, pdb_chain_init, pdb_chain_final):
    pdb_init, chain_init = pdb_chain_init.split("_")
    pdb_final, chain_final = pdb_chain_final.split("_")

    cif_file_init = f"{conformation_dir}/{pdb_init}.cif"
    cif_file_final = f"{conformation_dir}/{pdb_final}.cif"

    df_init = parse_cif_biopandas(cif_file_init, chain_id=chain_init)
    df_final = parse_cif_biopandas(cif_file_final, chain_id=chain_final)

    residues_init = get_residue_dict(df_init)
    residues_final = get_residue_dict(df_final)

    return residues_init, residues_final


def gather_ca_coords_pair(residues_init, residues_final, align: bool = False):
    ca_coords_init = np.asarray([
        gather_ca_coords(residue_info) for res_id, residue_info in residues_init.items()
    ])

    ca_coords_final = np.asarray([
        gather_ca_coords(residue_info) for res_id, residue_info in residues_final.items()
    ])

    if align:
        rot_mat, tr = rigid_transform_kabsch_3D(ca_coords_init.T, ca_coords_final.T)
        ca_coords_init_aligned = ( (rot_mat @ ca_coords_init.T) + tr ).T
        return ca_coords_init_aligned, ca_coords_final
    return ca_coords_init, ca_coords_final


def parse_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset", default="d3pm", type=str)

    # Filtering criteria
    parser.add_argument("--rmsd_tol", type=float, default=0.1, help="Predicted vs given RMSD diff")
    parser.add_argument("--rmsd_min", type=float, default=3.0, help="Min RMSD that we accept")
    
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    print(f"Arguments used in dataset prep: {args}")

    pdb_chains_filtered = []
    conformation_dir = f"{args.data_dir}/raw/{args.dataset}/conformations"
    
    conf_df = load_conformation_df(args)

    pdb_chains_all = conf_df[['Structure_apo', 'Structure_bound']].values.tolist()
    pdb_ids_file = f"{args.data_dir}/raw/{args.dataset}/conf_pairs_interim.txt"
    
    print(f"Writing intermediate ids to {pdb_ids_file}", flush=True)
    for idx, (pdb_chain_init, pdb_chain_final) in enumerate(pdb_chains_all):
        try:
            rmsd_true = conf_df.loc[idx, "RMSD_overall"]
            if rmsd_true < args.rmsd_min:
                continue

            print(f"PDB Chains: Init={pdb_chain_init}, Final={pdb_chain_final}", flush=True)    
            residues_init, residues_final = load_cif_and_residues(
                                    conformation_dir=conformation_dir,
                                    pdb_chain_init=pdb_chain_init,
                                    pdb_chain_final=pdb_chain_final
                                )

            print(f"Residue Count Raw: Init={len(residues_init)}, Final={len(residues_final)}", flush=True)

            residues_init, residues_final = align_residue_dicts(
                    residues_A=residues_init, residues_B=residues_final
            )

            print(f"Residue Count Common: Init={len(residues_init)}, Final={len(residues_final)}", flush=True)

            ca_coords_init_aligned, ca_coords_final = \
                    gather_ca_coords_pair(residues_init, residues_final, align=True)

            rmsd_post_align = rmsd(ca_coords_init_aligned, ca_coords_final)
            print(f"RMSD Aligned: {np.round(rmsd_post_align, 4)}, RMSD True: {np.round(rmsd_true, 4)}")

            rmsd_check_passed = rmsd_check(args, rmsd_post_align, rmsd_true)
            if rmsd_check_passed:
                print(f"Accepted by RMSD check, diff<{args.rmsd_tol}, rmsd>{args.rmsd_min}", flush=True)
                pdb_chains_filtered.append((pdb_chain_init, pdb_chain_final))
                with open(pdb_ids_file, "a") as f:
                    f.write(f"{pdb_chain_init},{pdb_chain_final}\n")

            print(flush=True)

        except Exception as e:
            print(e, flush=True)
            traceback.print_exc()
            continue

    pdb_ids_file = f"{args.data_dir}/raw/{args.dataset}/conf_pairs.txt"
    print(f"Writing filtered ids to {pdb_ids_file}", flush=True)

    with open(pdb_ids_file, "w") as f:
        for pdb_chain_init, pdb_chain_final in pdb_chains_filtered:
            f.write(f"{pdb_chain_init},{pdb_chain_final}\n")

if __name__ == "__main__":
    main()
