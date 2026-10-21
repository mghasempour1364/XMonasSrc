X-MONAS: Explainable Multi-Objective Neuro-Symbolic Feature Selection Framework

**Overview**
This repository contains the source code for the **X-MONAS** framework, an uncertainty-aware, neuro-symbolic, multi-objective framework for adaptive feature selection and disease diagnosis. The code implements the methodology described in the manuscript: *"An Explainable Multi-Objective Neuro-Symbolic Feature Selection Framework Using Self-Adaptive Hybrid Swarm Intelligence for Robust Disease Diagnosis"*.

## System Requirements
- **MATLAB R2015a** or later.
- **Statistics and Machine Learning Toolbox** (for `fitcknn` and evaluation metrics).
- **Deep Learning Toolbox** (optional, for the neural network in `XMONAS_NS_MOFS.m`).
- **Parallel Computing Toolbox** (optional, for faster execution).

## Folder Structure
- `Main.m`: The main execution script. Run this file to reproduce the results.
- `Datasets/`: Contains the dataset(s) in `.xlsx` or `.csv` format (e.g., `2_ILPD.xlsx`).
- `preprocessing/`: Functions for the Robust Medical Data Preprocessing (RMDP) layer (KNN imputation, Z-score outlier correction, Min-Max normalization, PCA).
- `xmonas/`: Core X-MONAS modules (NS-MOFS, SA-HSI, EDOF, UAFE, FDDS).
- `baselines/`: Baseline methods (MOGA, MOHHO, MOGAHHO) for comparison.
- `evaluation/`: Functions for calculating metrics (Balanced Accuracy, MCC, BM, MK, Precision@Top-N).
- `utils/`: Utility functions (data loading, cross-validation split).
- `results/`: Output folder for figures and saved results.

## How to Run
1.  Clone this repository to your local machine.
2.  Open MATLAB and navigate to the project root folder.
3.  Open and run `Main.m`.
4.  The script will load the dataset, perform 5-fold cross-validation, run all methods, and display the results in the Command Window.
5.  Figures (Runtime and Precision@Top-3) will be saved in the `results/` folder.

## Citation
If you use this code in your research, please cite our manuscript (details will be provided upon acceptance).

## Contact
For questions, please contact: me.ghasempour@iau.ac.ir
