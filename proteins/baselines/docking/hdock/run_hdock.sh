#!/bin/bash

module load fftw/3.3.9

LIGAND_DIR="inference/files/db5/test_inputs"
RECEPTOR_DIR="inference/files/db5/test_complexes"

HDOCK_OUT_DIR="hdock/db5/outputs"
HDOCK_PDB_DIR="inference/methods/db5/hdock_results"
mkdir -p ${HDOCK_OUT_DIR}
mkdir -p ${HDOCK_PDB_DIR}

PDB_IDS="$(cat ${PROJ_DIR}/sbalign/data/raw/db5/test.txt)"


echo "`lscpu | grep "Model name"`"
echo "Start Time = `date +%T`"

for PDB_ID in ${PDB_IDS}
do
    LIG_FILE="${LIGAND_DIR}/${PDB_ID}_l_u_rigid.pdb"
    REC_FILE="${RECEPTOR_DIR}/${PDB_ID}_r_b_COMPLEX.pdb"
    
    echo "Running inference for pdb id ${PDB_ID}"
    echo "Ligand file located at ${LIG_FILE}"
    echo "Receptor file located at ${REC_FILE}"
    echo " "
    
    ./bin/hdock "${REC_FILE}" "${LIG_FILE}" -out "${PDB_ID}.out"
    mv ${PDB_ID}.out "${HDOCK_OUT_DIR}/"
    ./bin/createpl "${HDOCK_OUT_DIR}/${PDB_ID}.out" "${HDOCK_PDB_DIR}/${PDB_ID}_l_b_HDOCK.pdb" -nmax 1

done

echo "End Time = `date +%T`"



