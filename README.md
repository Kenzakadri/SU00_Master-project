# SU00_Master-project

Drafts and scripts written during my Master 2 in Systems Biology (Sorbonne Université), from single-neuron modelling to machine-learning analyses of biomedical data.

The core of the repository is a series of **TMEs** (*Travaux sur Machine Encadrés*, supervised lab sessions) used to learn machine learning step by step, followed by larger projects that apply those methods to real datasets. Most code is in Python (Jupyter notebooks, mostly `scikit-learn`), plus one R/Bioconductor project on gene regulatory networks.

---

## Repository structure

```
Data/     datasets used in the notebooks (Bayesian network data, archived datasets)
Plot/     figures exported from the notebooks (iris plots, Bayesian network graphs)
Script/   all notebooks and scripts (TMEs + projects)
```

> Some input files read by the notebooks (e.g. `patients_data.txt`, `Golub_X`, `Breast.txt`, `BreastDiagnostic.txt`, `data.txt`, `train_data.txt`) are not included in the repo and need to be placed next to the scripts to re-run them.

---

## Machine learning TMEs — overview

| TME | File | Topic | Methods | Data |
|---|---|---|---|---|
| TME 3 | `Script/TME3.py` | Supervised classification with trees | Decision tree, random forest, feature importance, comparison with the clinical DiaRem score | Type 2 diabetes remission after gastric bypass (age, HbA1c, insulin, other drugs) |
| TME 5 | `Script/TME5.ipynb` | Logistic regression & overfitting | Logistic regression (sklearn and implemented from scratch by gradient optimisation), decision boundaries, train/test split, K-fold and repeated K-fold cross-validation | Simulated data (`make_classification`, `make_blobs`, `make_moons`), breast cancer diagnostic data |
| TME 6 | `Script/TME6K.ipynb` | Support Vector Machines | SVM with linear / RBF / polynomial kernels, hyper-parameter search (C, kernel) with K-fold CV, accuracy boxplots, role of support vectors | 3 simulated datasets (classification, 3-class blobs, moons) |
| TME 8 | `Script/TME8.ipynb` | Feature selection & model selection | Variance threshold, univariate selection (`SelectFdr`), L1-based selection (Lasso, `LinearSVC`), Elastic Net | Golub leukaemia gene expression, breast cancer dataset |
| TME 9 | `Script/Tme09.ipynb` | Bayesian networks for clinical data | Building a BN by hand and running inference (pyAgrum), structure learning from data (Greedy hill climbing, Local search), dynamic Bayesian networks | Diabetes remission toy model, SPLEX environmental & host data, longitudinal data (HbA1c, glycaemia, weight, status at 0/1/5 years) |

### Key learning points
- **TME 3** – a tree evaluated on its own training set gives 0 % error: why we need held-out data; random forests to rank variable importance.
- **TME 5** – how logistic regression is fitted, and how cross-validation gives an honest estimate of accuracy.
- **TME 6** – choosing a kernel and regularisation parameter `C` depending on the shape of the data (linear vs non-linear boundaries).
- **TME 8** – reducing high-dimensional data (thousands of genes, few patients) before classification.
- **TME 9** – probabilistic graphical models: conditional probabilities, inference, and learning network structure from data. Network graphs are saved in `Plot/network_bn_*.pdf`.

---

## Projects

### Life Sciences Data Exploration — `Script/ProjetSPLEX.ipynb`
"Propose and develop an optimal method": classification of a train/test dataset by comparing algorithms (logistic regression, KNN, SVM, LDA, naive Bayes, Elastic Net…), feature selection (`SelectKBest`, `SelectFromModel`), correlation analysis, UMAP visualisation and cross-validation.

### Single-cell RNA-seq analysis (Advanced Physiology) — `Script/AdSP-Projet.ipynb`, `Script/Projet final.ipynb`, `Script/Projet_final_final.ipynb`
Identification of cell sub-populations from scRNA-seq data:
1. Quality control (UMIs per cell, genes per cell, filtering cells with 12 000–100 000 UMIs and < 30 % mitochondrial reads)
2. Normalisation (`ln(x / UMI_cell × median + 1)`)
3. Selection of the 1 500 most variable genes (coefficient of variation)
4. Dimensionality reduction (PCA, UMAP)
5. Clustering (K-means, hierarchical/agglomerative) and marker genes

`Projet_final_final.ipynb` is the final version; the other two are earlier drafts.

---

## Gene regulatory networks (R) — `Script/projet_resys.R`

Master regulator analysis of osteosarcoma metastasis using public transcriptomic data (GEO **GSE21257**):
1. Data retrieval from GEO (`GEOquery`) and sample-level quality check
2. Inference of a transcriptional regulatory network with `RTN` (mutual information, 1 000 permutations, 100 bootstraps, DPI filtering)
3. Differential expression between metastatic and non-metastatic groups with `limma` (FDR correction)
4. Master Regulator Analysis and two-tailed GSEA on regulons to find transcription factors driving the phenotype (e.g. IRF8, VAV1, VENTX)
5. Network visualisation with `RedeR`

Slow steps can be skipped by reloading saved `.RData` checkpoints (`use_checkpoints <- TRUE`; checkpoints not included).

---

## Neuron modelling (tutorials)

| File | Content |
|---|---|
| `Script/Brian2_tutorial.ipynb` | Introduction to the Brian2 spiking-network simulator: units, neuron equations, spikes, monitors |
| `Script/First_data_analysis.ipynb` | NEURON simulator: building a soma + dendrites model with Hodgkin–Huxley and passive channels, alpha synapses |
| `Script/Neuron_tutorial.ipynb` | NEURON setup (empty starter) |

---

## Main dependencies

**Python:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `statsmodels`, `umap-learn`, `scanpy`, `scprep`, `anndata`, `pyAgrum`, `graphviz`, `brian2`, `neuron`

**R / Bioconductor:** `Biobase`, `GEOquery`, `limma`, `RTN`, `snow`, `RedeR`
