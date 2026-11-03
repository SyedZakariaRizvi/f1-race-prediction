import argparse
import warnings
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw",
                        help="Directory containing the six original Ergast CSVs.")
    parser.add_argument("--output-dir", type=Path, default=root / "results",
                        help="Destination for metrics.csv and four plots.")
    parser.add_argument("--interim-dir", type=Path, default=root / "data" / "interim",
                        help="Destination for the merged master dataset.")
    args = parser.parse_args()
    # Preserve the notebook's filter for LightGBM's array feature-name warning.
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    # Set a noninteractive backend before importing plotting code.
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd

    from src.data import build_master_dataset, clean_dataset, group_rare_categories, split_dataset
    from src.evaluate import best_model_details, compare_models
    from src.models import ranking_models, regression_models
    from src.plots import plot_data, plot_ranking_quality
    from src.preprocessing import build_preprocessor, categorical_features
    from src.train import (
        prepare_ranking_data, train_rankers, train_regressors,
        tune_elastic_net, tune_ranker,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.interim_dir.mkdir(parents=True, exist_ok=True)

    print("Building the master dataset from six Ergast tables...", flush=True)
    df = build_master_dataset(args.data_dir)
    master_path = args.interim_dir / "f1_master_dataset.csv"
    df.to_csv(master_path, index=False)
    # Preserve the notebook's CSV round-trip before cleaning and training.
    df = pd.read_csv(master_path)
    df_clean = clean_dataset(df)
    print(f"Master dataset: {df.shape}; modern-era data: {df_clean.shape}", flush=True)
    plot_data(df_clean, args.output_dir)
    df_clean = group_rare_categories(df_clean, categorical_features, top_n=10)
    train_df, test_df, X_train, X_test, y_train, y_test = split_dataset(df_clean)
    preprocessor = build_preprocessor()

    print("Training five regression baselines...", flush=True)
    predictions, _, regression_metrics = train_regressors(
        regression_models(), preprocessor, X_train, y_train, X_test, y_test, test_df,
    )
    print(regression_metrics.to_string(index=False), flush=True)

    print("Tuning Elastic Net with race-grouped cross-validation...", flush=True)
    _, pred_enet_tuned = tune_elastic_net(
        preprocessor, X_train, y_train, X_test, y_test, train_df,
    )
    predictions["Elastic Net (tuned)"] = (pred_enet_tuned, test_df, False)

    print("Training XGBoost and LightGBM rankers...", flush=True)
    _, test_ltr, _, X_tr_ltr, X_te_ltr, y_tr_rel, qid_tr, group_tr = prepare_ranking_data(
        preprocessor, train_df, test_df, X_train,
    )
    ltr_models = ranking_models()
    ranker_predictions, _, best_ltr_name = train_rankers(
        ltr_models, X_tr_ltr, y_tr_rel, group_tr, X_te_ltr, test_ltr,
    )
    predictions.update(ranker_predictions)

    print(f"Tuning {best_ltr_name} with race-grouped cross-validation...", flush=True)
    tuned_name, _, tuned_scores, _ = tune_ranker(
        ltr_models, best_ltr_name, X_tr_ltr, y_tr_rel, qid_tr, group_tr, X_te_ltr,
    )
    predictions[tuned_name] = (tuned_scores, test_ltr, True)

    comparison = compare_models(predictions)
    comparison.to_csv(args.output_dir / "metrics.csv", index=False)
    best_name, eval_df = best_model_details(comparison, predictions)
    plot_ranking_quality(eval_df, args.output_dir)
    print(comparison.round(4).to_string(index=False))
    print(f"Highest 2024 NDCG@10: {best_name}")
    print(f"Saved metrics and four plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
