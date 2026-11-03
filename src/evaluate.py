import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, ndcg_score

MAX_POS = 25


def evaluate_f1_predictions(model_name, raw_scores, test_df_ref, higher_score_is_better=True):
    """
    Race-by-race evaluation of any regression or Learning-to-Rank model on the test set.
    - Regression models (predict positionOrder):  higher_score_is_better=False
    - LTR models (predict a relevance score):     higher_score_is_better=True
    raw_scores must be in the same row order as test_df_ref.
    """
    df_eval = test_df_ref[["raceId", "driverId", "positionOrder"]].copy()
    df_eval["score"] = raw_scores if higher_score_is_better else -np.asarray(raw_scores)
    df_eval["true_rel"] = np.clip(MAX_POS - df_eval["positionOrder"], 0, None)

    ndcg_3, ndcg_10, ndcg_all, spearman, winner_hits, podium_overlap, top5_overlap = [], [], [], [], [], [], []
    all_true_pos, all_pred_pos = [], []

    for _, group in df_eval.groupby("raceId", sort=False):
        group = group.sort_values("score", ascending=False).reset_index(drop=True)
        group["pred_position"] = np.arange(1, len(group) + 1)

        true_rel = group["true_rel"].to_numpy().reshape(1, -1)
        scores = group["score"].to_numpy().reshape(1, -1)
        all_true_pos.extend(group["positionOrder"])
        all_pred_pos.extend(group["pred_position"])

        ndcg_3.append(ndcg_score(true_rel, scores, k=3))
        ndcg_10.append(ndcg_score(true_rel, scores, k=10))
        ndcg_all.append(ndcg_score(true_rel, scores))
        spearman.append(spearmanr(group["positionOrder"], group["pred_position"])[0])
        winner_hits.append(int(group.loc[0, "positionOrder"] == 1))
        true_podium = set(group.loc[group["positionOrder"] <= 3, "driverId"])
        pred_podium = set(group.loc[group["pred_position"] <= 3, "driverId"])
        podium_overlap.append(len(true_podium & pred_podium) / 3)
        true_top5 = set(group.loc[group["positionOrder"] <= 5, "driverId"])
        pred_top5 = set(group.loc[group["pred_position"] <= 5, "driverId"])
        top5_overlap.append(len(true_top5 & pred_top5) / 5)

    return {
        "Model": model_name,
        "NDCG@3": np.mean(ndcg_3),
        "NDCG@10": np.mean(ndcg_10),
        "NDCG@All": np.mean(ndcg_all),
        "Spearman_Rho": np.mean(spearman),
        "Winner_Acc (%)": np.mean(winner_hits) * 100,
        "Podium_Hit (%)": np.mean(podium_overlap) * 100,
        "Top5_Hit (%)": np.mean(top5_overlap) * 100,
        "Rank_MAE": mean_absolute_error(all_true_pos, all_pred_pos),
        "Rank_RMSE": root_mean_squared_error(all_true_pos, all_pred_pos),
    }


def mean_ndcg(y_rel, scores, sizes, k=10):
    """Mean per-race NDCG@k."""
    out, start = [], 0
    for g in sizes:
        out.append(ndcg_score([y_rel[start:start + g]], [scores[start:start + g]], k=k))
        start += g
    return np.mean(out)


def compare_models(predictions):
    """Compare every model using the original per-race metrics."""
    comparison = pd.DataFrame([
        evaluate_f1_predictions(name, scores, ref, higher)
        for name, (scores, ref, higher) in predictions.items()
    ]).sort_values("NDCG@10", ascending=False).reset_index(drop=True)

    return comparison


def best_model_details(comparison, predictions):
    """Construct within-race positions for the highest test NDCG@10 model."""
    best_name = comparison.loc[0, "Model"]
    scores, ref, higher = predictions[best_name]

    eval_df = ref[["raceId", "race_round", "starting_grid", "qualifying_position", "positionOrder"]].copy()
    eval_df["score"] = scores if higher else -np.asarray(scores)
    eval_df["predicted_position"] = eval_df.groupby("raceId")["score"].rank(ascending=False, method="first").astype(int)

    return best_name, eval_df
