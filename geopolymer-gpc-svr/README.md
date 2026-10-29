# Modeling the Compressive Strength of Bacteria-induced Metakaolin-Based Geopolymer Concrete

Undergraduate research project applying Support Vector Regression (SVR) to predict the compressive strength of bacteria-induced metakaolin-based geopolymer concrete (GPC) from mix design variables, and investigating the material's water absorption performance.

## Background

Geopolymer concrete is an alkali-activated alternative to ordinary Portland cement concrete, offering reduced carbon footprint by using calcined clay (metakaolin) as a binder instead of clinker. This project incorporates *Bacillus subtilis* bacteria into the mix to investigate impart on compressive strength, and uses machine learning (SVR) to model how mix design parameters relate to 28-day compressive strength.

The specific objectives were to:
1. Characterize materials used in the production of the bacteria-induced Metakaolin-based Geopolymer concrete.
2. Develop suitable mix ratios for the Geopolymer concrete.
3. Formulate a predictive model for compressive strength using a Support Vector Regression (SVR) machine learning approach.
4. Measure the water absorption rate of specimens after 24-hour water immersion.

## Methodology

- **Raw material characterization:** Metakaolin composition verified via X-ray Fluorescence (XRF) and X-ray Diffraction (XRD) analysis prior to mix design. Particle size gradation, bulk density test and specific gravity test carried out on the fine and coarse aggregate used.
- **Mix design variables:** NaOH molarity (8–14M), Na₂SiO₃:NaOH activator ratio (1.5–2.5), *Bacillus subtilis* dosage (%), activator-to-binder ratio, and curing temperature.
- **Modeling approach:** Support Vector Regression (SVR) with a radial basis function (RBF) kernel, implemented in scikit-learn. Hyperparameters (C, epsilon, gamma) were tuned via 5-fold cross-validated grid search.
- **Correlation analysis:** Pearson correlation was used to screen each mix variable's linear association with compressive strength prior to modeling, following documented exclusion of a small number of specimens with casting-related physical defects.
- **Validation:** Model performance evaluated via R², MAE, MSE, RMSE, and MAPE on a held-out test set.

## Repository Structure

```
├── data/
│   └── gpc_datset.csv                  # Experimental mix design and strength dataset
├── notebooks/
│   └── svr_gpc_model.ipynb            # Main analysis notebook (data loading → modeling → evaluation)
├── src/
│   └── svr_gpc_compressive_strength.py # Standalone script version of the SVR pipeline
├── results/
│   ├── predicted_vs_actual.png        # Model fit visualization (test set)
│   ├── feature_importance.png         # Permutation feature importance
│   ├── support_vectors.csv            # Exported SVR support vectors and dual coefficients
│   └── scaler_parameters.csv          # StandardScaler mean/std for reconstructing the model equation
├── docs/
│   └── Results_and_Discussion.docx    # Full written results and discussion
├── requirements.txt
└── README.md
```

## Key Results

| Dataset | R² | MAE (MPa) | RMSE (MPa) | MAPE (%) |
|---|---|---|---|---|
| Training set | 0.2017 | 0.5312 | 0.6243 | 2.17 |
| Test set | −0.0428 | 0.6886 | 0.8132 | 2.81 |

**Correlation with compressive strength** (after exclusion of specimens with documented casting defects):

| Variable | r |
|---|---|
| Activator Ratio (Na₂SiO₃:NaOH) | 0.311 |
| Temperature | 0.144 |
| *Bacillus subtilis* Dosage | 0.114 |
| Liquid/Binder Ratio | 0.040 |
| NaOH Molarity | −0.026 |

**Interpretation:** Within the mix design ranges tested, the activator ratio showed the clearest (though moderate) relationship with compressive strength. The SVR model captured a modest pattern in training data (R² = 0.20) but did not generalize reliably to unseen specimens (test R² = −0.04), most likely due to the limited dataset size (70 specimens), weak individual variable correlations, and a relatively narrow range of measured strength values (~21–26 MPa). This is reported as a genuine finding rather than a strong predictive result — see the full discussion in `docs/Results_and_Discussion.docx` for a detailed treatment of why this outcome occurred and what it suggests for future work.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Gideon0323/metakaolin-gpc-compressive-strength.git
   cd metakaolin-gpc-compressive-strength
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook notebook/my_svr_gpc_model.ipynb
   ```
   or run the standalone script:
   ```bash
   python src/svr_gpc_compressive_strength.py
   ```

## Tech Stack

- Python 3
- scikit-learn (SVR, GridSearchCV, permutation importance)
- pandas / numpy
- matplotlib
- Jupyter Notebook

## Limitations and Honest Reporting

A small number of specimens (documented as having physical casting defects in lab records) were excluded from the correlation analysis and final model training, with this exclusion stated explicitly rather than applied silently. Even after this exclusion, the model's test-set performance indicates the mix design variables tested do not yet support strong predictive modeling of compressive strength with this dataset size. This is discussed transparently in the full report as a basis for future work (larger datasets, narrower parameter ranges around the activator ratio, and tighter control of casting/curing conditions) rather than presented as a limitation to minimize.

## Author

**Chibuike, Chinazaekpere Gideon**
Final-year undergraduate researcher
Federal University of Technology, Owerri.
gideonc430@gmail.com

## License

"This project is licensed under the terms of the MIT license." 