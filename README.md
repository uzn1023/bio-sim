# bio-sim

This repository contains Python implementations of bacterial growth models inspired by the article **"Kinetics of Bacterial Adaptation, Growth, and Death at Didecyldimethylammonium Chloride sub-MIC Concentrations"** (Pedreira et al., 2022). Two models are provided:

* `simulate_ddac.py` – demonstrates logistic and mechanistic models using example parameters.
* `fit_ddac.py` – fits either model to OD vs. time data at different disinfectant concentrations.

The original dataset from the article is hosted on Zenodo (doi:10.5281/zenodo.5167910) and cannot be downloaded automatically. A small example dataset is included under `data/example_data.csv` to illustrate usage. Replace this file with the full dataset to reproduce the published results.

## Running simulations

```bash
python simulate_ddac.py
```

This command generates example growth curves for both models.

## Parameter fitting

```bash
python fit_ddac.py --data data/example_data.csv --model logistic
```

Use `--model mechanistic` to fit the mechanistic model. The script prints the estimated parameters and plots the fitted curves.
