# Aligned Diffusion Schrödinger Bridges

(Under Construction and Subject to Change)

This is the official [PyTorch](https://pytorch.org/) implementation for _SBAlign_ ([Somnath et al. 2023](https://arxiv.org/abs/2302.11419))

## Installation

### Environment
To install the conda environment and necessary packages, run the following command

```bash
./build_env.sh
```
The installation should work on Linux, Mac and M1/M2 Mac.

## Experiments

### A. Toy (spiral and T datasets)

Training and visualization code can be found in `toy/toy_experiments.ipynb`.

Trained models (and corresponding datasets) can be loaded with:

    exp = AlignExperiment.load(run_name)

To sample trajectories, call:

    exp.sample(...)


### B. Cell Differentiation Processes

Training and visualization code can be found in `cells/cells_experiment.ipynb`.

Trained models (and corresponding datasets) can be loaded with:

    exp = AlignExperiment.load(run_name)

To sample trajectories, call:

    exp.sample(...)

Additional remarks:

- Trajectories sampled from (i) the baseline, (ii) SBalign and (iii) the baseline + drift (from SBalign) are stored in `cells/results/`.
- Wasserstein distances between end distributions can be computed with `cells/wasserstein_metric.ipynb`.

### C. Protein Conformational Changes

In this task, we are interested in modeling conformational changes between unbound and bound states of the protein.

#### Datasets
Datasets are organized under `data/`. Raw and processed datasets are stored under `data/raw` and `data/processed` respectively.

For this task, we use the `D3PM` dataset, (`data/raw/d3pm` and `data/processed/d3pm`)

#### Preprocessing

##### Downloading structures
The file with PDB IDs of ligand-free and ligand-bound structures can be downloaded from [here](http://www.d3pharma.com/D3PM/overall_apo_com.php).
Rename this file to `d3pm.xlsx` and place under `data/raw/d3pm`.

The structures corresponding to PDB IDs can be downloaded by following the instructions on the [here](https://www.rcsb.org/downloads).
For this task, we downloaded the `.cif` files, which were saved to `data/raw/d3pm/conformations`

##### Dataset Preparation

To filter the acceptable structures (based on criteria defined in the paper), run
```
python scripts/conf/prepare_dataset.py --data_dir data --dataset d3pm
```

The dataset can then be preprocessed by running the following command:
```
python scripts/conf/preprocess.py --center_conformations --resolution c_alpha
```

The raw and processed D3PM datasets can be found at [zenodo](https://zenodo.org/record/8066711)

#### Training & Evaluation

To train the model, run the following command:

```
python scripts/conf/train.py --config ${PATH_TO_CONFIG}.yml
```

To evaluate the trained model, run the following command:
```
python scripts/conf/evaluate.py --data_dir data --log_dir logs --run_name ${RUN_NAME} \
    --model_name ${MODEL_NAME} --method sbalign --inference_steps 10 --n_samples 10
```

For the model used in the paper, the configuration file used can be found under `reproducibility/conf/train.yml`
The corresponding trained model can be found under `reproducibility/conf/model.pt`. To evaluate this model, run:

```
python scripts/conf/evaluate.py --data_dir data --log_dir reproducibility --run_name conf \
    --model_name model.pt --method sbalign --inference_steps 100 --n_samples 10
```

### D. Rigid-Protein Docking

In this task, we are interested in learning a stochastic process that best orients the ligand protein relative to the receptor protein.

#### Datasets

For this task, we use the `DB5.5` dataset, (`data/raw/db5` and `data/processed/db5`)

#### Preprocessing

##### Downloading structures

The structures can be downloaded following the links listed on the [EquiDock repo](https://github.com/octavian-ganea/equidock_public)
The complex structures are stored under `data/raw/db5/complexes` and the train/valid/test splits are gathered into `data/raw/db5/splits.json`.

##### Dataset Preparation

The dataset can then be preprocessed by running the following command:
```
python scripts/docking/preprocess.py --resolution c_alpha
```

The raw and processed DB5.5 datasets can be found at [zenodo](https://zenodo.org/record/8066711)

#### Training

To train the model, run the following command:

```
python scripts/docking/train.py --config ${PATH_TO_CONFIG}.yml
```

## License

This project is licensed under the MIT-License. Please see [LICENSE.md](https://github.com/IBM/aligned_diffusion_bridges/blob/main/LICENSE.md) for more details.

## Reference

If you find our code useful, please cite our paper:

```
@article{somnath2023aligned,
  title={Aligned Diffusion Schr$\backslash$" odinger Bridges},
  author={Somnath, Vignesh Ram and Pariset, Matteo and Hsieh, Ya-Ping and Martinez, Maria Rodriguez and Krause, Andreas and Bunne, Charlotte},
  journal={arXiv preprint arXiv:2302.11419},
  year={2023}
}
```
## Contact

If you have any questions about the code, or want to report a bug, or need help interpreting an error message, please raise a GitHub issue.
