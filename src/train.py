import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, ParameterGrid
from .evaluate import MAX_POS, evaluate_f1_predictions, mean_ndcg


def train_regressors(models, preprocessor, X_train, y_train, X_test, y_test, test_df):
    """Return predictions, fitted pipelines, and raw regression MAE/RMSE."""
    predictions, fitted = {}, {}
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", model)]).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        predictions[name] = (pred, test_df, False)
        rows.append({"Model": name,
                     "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, pred))})
    return predictions, fitted, pd.DataFrame(rows).round(3)


def tune_elastic_net(preprocessor, X_train, y_train, X_test, y_test, train_df):
    """Run the original 30-setting, five-fold race-grouped search."""
    enet_pipe = Pipeline([("preprocessor", clone(preprocessor)),
                          ("model", ElasticNet(max_iter=10000, random_state=42))])

    param_grid = {
        "model__alpha": [0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }

    enet_grid = GridSearchCV(enet_pipe, param_grid, scoring="neg_mean_absolute_error",
                             cv=GroupKFold(n_splits=5), n_jobs=-1)
    enet_grid.fit(X_train, y_train, groups=train_df["raceId"].to_numpy())

    pred_enet_tuned = enet_grid.predict(X_test)

    print("Best params :", enet_grid.best_params_)
    print(f"CV MAE      : {-enet_grid.best_score_:.3f}")
    print(f"Test MAE    : {mean_absolute_error(y_test, pred_enet_tuned):.3f}")
    print(f"Test RMSE   : {np.sqrt(mean_squared_error(y_test, pred_enet_tuned)):.3f}")
    return enet_grid, pred_enet_tuned


def group_sizes(qid):
    """Number of drivers per race, in order of appearance (data must be contiguous by race)."""
    return pd.Series(qid).groupby(qid, sort=False).size().to_numpy()


def prepare_ranking_data(preprocessor, train_df, test_df, X_train):
    """Sort race groups and transform features with the original shared preprocessor."""
    sort_cols = ["season", "race_round", "raceId"]
    train_ltr = train_df.sort_values(sort_cols).reset_index(drop=True)
    test_ltr = test_df.sort_values(sort_cols).reset_index(drop=True)

    ltr_prep = clone(preprocessor)
    X_tr_ltr = np.asarray(ltr_prep.fit_transform(train_ltr[X_train.columns], train_ltr["positionOrder"]))
    X_te_ltr = np.asarray(ltr_prep.transform(test_ltr[X_train.columns]))

    y_tr_rel = np.clip(MAX_POS - train_ltr["positionOrder"].to_numpy(), 0, None).astype(int)
    qid_tr = train_ltr["raceId"].to_numpy()

    group_tr = group_sizes(qid_tr)
    print(f"Train: {X_tr_ltr.shape[0]} rows / {len(group_tr)} races | Test: {X_te_ltr.shape[0]} rows / {test_ltr['raceId'].nunique()} races")
    return train_ltr, test_ltr, ltr_prep, X_tr_ltr, X_te_ltr, y_tr_rel, qid_tr, group_tr


def train_rankers(ltr_models, X_tr_ltr, y_tr_rel, group_tr, X_te_ltr, test_ltr):
    """Fit both rankers and select the one with the highest 2024 NDCG@10."""
    predictions = {}
    basic_ltr = []
    for name, (Ranker, params) in ltr_models.items():
        ranker = Ranker(**params).fit(X_tr_ltr, y_tr_rel, group=group_tr)
        scores = ranker.predict(X_te_ltr)
        predictions[name] = (scores, test_ltr, True)
        basic_ltr.append(evaluate_f1_predictions(name, scores, test_ltr, True))

    basic_ltr = pd.DataFrame(basic_ltr).round(4)
    best_ltr_name = basic_ltr.sort_values("NDCG@10", ascending=False).iloc[0]["Model"]
    print("Best basic ranker:", best_ltr_name)
    return predictions, basic_ltr, best_ltr_name


def tune_ranker(ltr_models, best_ltr_name, X_tr_ltr, y_tr_rel, qid_tr, group_tr, X_te_ltr):
    """Run the original 12-setting, four-fold race-grouped ranking search."""
    Ranker, base_params = ltr_models[best_ltr_name]
    param_grid = {"n_estimators": [100, 300, 500], "learning_rate": [0.03, 0.1], "max_depth": [3, 5]}

    cv = GroupKFold(n_splits=4)
    cv_results = []
    for params in ParameterGrid(param_grid):
        fold_scores = []
        for tr, va in cv.split(X_tr_ltr, groups=qid_tr):
            model = Ranker(**{**base_params, **params}).fit(X_tr_ltr[tr], y_tr_rel[tr], group=group_sizes(qid_tr[tr]))
            fold_scores.append(mean_ndcg(y_tr_rel[va], model.predict(X_tr_ltr[va]), group_sizes(qid_tr[va])))
        cv_results.append({**params, "cv_ndcg@10": np.mean(fold_scores)})

    cv_results = pd.DataFrame(cv_results).sort_values("cv_ndcg@10", ascending=False)
    best_params = cv_results.iloc[0].drop("cv_ndcg@10").to_dict()
    best_params = {k: (int(v) if k != "learning_rate" else float(v)) for k, v in best_params.items()}
    print("Best params:", best_params)

    tuned_ranker = Ranker(**{**base_params, **best_params}).fit(X_tr_ltr, y_tr_rel, group=group_tr)
    tuned_name = f"{best_ltr_name} (tuned)"
    tuned_scores = tuned_ranker.predict(X_te_ltr)

    return tuned_name, tuned_ranker, tuned_scores, cv_results
