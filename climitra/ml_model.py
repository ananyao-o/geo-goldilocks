import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

def minmax(x: pd.Series) -> pd.Series:
    a, b = x.min(), x.max()
    if b == a:
        return pd.Series([0.0]*len(x), index=x.index)
    return (x - a) / (b - a)

def make_features(zones: pd.DataFrame, avg_road: pd.Series) -> pd.DataFrame:
    df = zones.copy()
    df["avg_road_distance"] = avg_road.values
    # elevation deviation from median
    med = df["elevation"].median()
    df["elev_dev"] = (df["elevation"] - med).abs()
    return df

def initial_target(df: pd.DataFrame) -> pd.Series:
    # assuming heuristic prior targets (can be replaced by real labels later)
    s = minmax(df["slope"])            
    r = minmax(df["avg_road_distance"])
    e = minmax(df["elev_dev"])         
    y = 0.5*s + 0.35*r + 0.15*e
    return y

def train_model(feat_df: pd.DataFrame, random_state: int = 42) -> Pipeline:
    X = feat_df[["slope","avg_road_distance","elev_dev","land_type"]]
    y = initial_target(feat_df)  # again, replace with real labels when available
    # categorical preprocessing
    cat = ["land_type"]
    num = ["slope","avg_road_distance","elev_dev"]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ]
    )
    model = GradientBoostingRegressor(random_state=random_state)
    pipe = Pipeline([("pre", pre), ("gbr", model)])
    pipe.fit(X, y)
    return pipe

def predict_scores(model: Pipeline, feat_df: pd.DataFrame) -> np.ndarray:
    X = feat_df[["slope","avg_road_distance","elev_dev","land_type"]]
    return model.predict(X)
