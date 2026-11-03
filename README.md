# Formula 1 Race Prediction with Learning-to-Rank

Can a model predict how an F1 race will finish? I merged six Ergast tables, built driver and constructor form features, and compared ordinary finishing-position regressors against LambdaMART rankers that learn to order drivers within each race. The best model, a tuned LightGBM ranker, scored an NDCG@10 of 0.934 on the 2024 season.

**Built with:** Python, scikit-learn, XGBoost, LightGBM, pandas, NumPy and Category Encoders.

## Results

Evaluated on the 24 races of 2024, the best model is tuned LightGBM LambdaMART:

| Metric | Result |
| --- | ---: |
| NDCG@10 | 0.9342 |
| Top-5 overlap | 80.83% |
| Winner prediction accuracy | 45.83% |

Full metrics for all nine models are in `results/metrics.csv` (the notebook shows them rounded to four decimals). Details on each artifact are in [results/README.md](results/README.md).

Please read the [limitations](#limitations) before trusting these scores. Several of them make the numbers look better than a clean setup would.

## Data and setup

The inputs are six Ergast files in `data/raw/`: `results.csv`, `races.csv`, `drivers.csv`, `constructors.csv`, `circuits.csv` and `qualifying.csv`. They cover results through 2024. The `*_2025_2026.csv` files and the other raw tables are in the repo but not used, and neither is `data/processed/model_dataset.csv`, which has a different schema.

Historical features are computed over the full history and then filtered to 2006 onward:

| Portion | Seasons | Driver–race records | Races |
| --- | --- | ---: | ---: |
| Training | 2006–2023 | 7,435 | 351 |
| Evaluation | 2024 | 479 | 24 |
| Total | 2006–2024 | 7,914 | 375 |

**Preprocessing:** target encoding for nationalities and circuit country, `log1p` plus scaling for skewed career stats, and scaling for the remaining numeric features.

**Models:**
- Five regressors: Linear Regression, Random Forest, Elastic Net, KNN and Gradient Boosting.
- Elastic Net is tuned with five-fold GroupKFold.
- Two rankers, XGBoost and LightGBM, learn race-level orderings. The better one is tuned with four-fold GroupKFold.
- Every fold keeps each race together.

## Running it

```bash
conda env create -f environment.yaml
conda activate f1-race-prediction
python run.py
```

This writes the merged dataset to `data/interim/f1_master_dataset.csv` and overwrites `results/metrics.csv` and the four PNGs. To keep the saved results, send the output somewhere else:

```bash
python run.py --output-dir results/new_run
```

You can also pass `--data-dir` and `--interim-dir`; see `python run.py --help`. Paths resolve relative to the script, so it works from any directory. The full run includes 150 Elastic Net CV fits and 48 ranker CV fits, plus the final and baseline fits, so expect it to take a while.

**Notebook:** launch `jupyter lab` from the project root (or from `notebooks/`), open `notebooks/f1_race_prediction.ipynb` and run the cells in order. It ships with the outputs of its latest run, and re-running refreshes them along with the files in `results/`.

The environment file gives compatible version ranges, not a lockfile. The original notebook never recorded exact versions, so library versions and platforms may shift your scores slightly.

## Project layout

```text
run.py                         # full experiment, raw data to results
environment.yaml
src/
├── data.py                    # merge tables, historical features, cleaning, split
├── preprocessing.py           # target encoding, log transforms, scaling
├── models.py                  # five regressors, two LambdaMART rankers
├── train.py                   # training and race-grouped hyperparameter search
├── evaluate.py                # per-race ranking metrics, model comparison
└── plots.py                   # the four figures
notebooks/
└── f1_race_prediction.ipynb   # step-by-step version of the experiment
data/
├── raw/                       # original input files
├── interim/                   # generated f1_master_dataset.csv
└── processed/                 # model_dataset.csv (unused)
results/                       # metrics.csv and four PNGs
```

## Limitations

I kept the original workflow as it was rather than quietly fixing things and changing the scores. These are the known problems:

- **Constructor history order.** Constructor features are computed after sorting by driver, year and round. Shifting in that order doesn't give true chronological constructor history, so future and same-race results can leak into these features.
- **Category grouping.** The top-ten category grouping uses all modern-era rows before the train/test split.
- **Preprocessing inside ranker CV.** Ranker features, including target encoding, are fitted once on all training years before cross-validation, so validation-fold labels influence preprocessing. Elastic Net's pipeline does refit within each fold.
- **No forward-time validation.** GroupKFold keeps races apart but isn't a forward-in-time split within the training years.
- **2024 is not a clean holdout.** I picked which ranker to tune using 2024 NDCG@10, and chose the final model on 2024 too.

The scores describe this workflow as it stands. Fixing these issues would change the model and need a fresh evaluation.

## Verification

The modular pipeline reproduces the notebook exactly. The merged dataset, cleaned inputs, split and transformed feature arrays all match, and a full run reproduces all nine saved model rows to four decimal places. Running the notebook's 26 code cells in order gives the same metrics as `run.py`.