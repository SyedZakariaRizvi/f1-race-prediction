from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from .evaluate import MAX_POS


def regression_models():
    """Return the five original regression estimators."""
    baseline_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    }

    extra_models = {
        "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=10),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
    }
    return {**baseline_models, **extra_models}


def ranking_models():
    """Return the XGBoost and LightGBM LambdaMART classes and settings."""
    shared = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)

    ltr_models = {
        "XGBoost LambdaMART": (xgb.XGBRanker, dict(shared, objective="rank:ndcg", ndcg_exp_gain=False)),
        "LightGBM LambdaMART": (lgb.LGBMRanker, dict(shared, objective="lambdarank", num_leaves=15, min_child_samples=30,
                                                     label_gain=[float(i ** 1.5) for i in range(MAX_POS + 1)],
                                                     n_jobs=-1, verbose=-1)),
    }

    return ltr_models
