import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from category_encoders import TargetEncoder

categorical_features = ['nationality', 'constructor_nationality', 'circuit_country']

skewed_features = [
    'career_wins', 'career_podiums', 'career_points', 'constructor_wins'
]

normal_numerical_features = [
    'starting_grid', 'qualifying_position', 'driver_age', 'career_starts',
    'avg_finish_before_race', 'recent_form', 'constructor_starts',
    'constructor_recent_form'
]


def build_preprocessor():
    """Build a fresh copy of the notebook preprocessing pipeline."""
    log_scale_transformer = Pipeline(steps=[
        ('log1p', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('skewed', log_scale_transformer, skewed_features),
            ('normal', StandardScaler(), normal_numerical_features),
            ('cat', TargetEncoder(min_samples_leaf=5, smoothing=10), categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor
