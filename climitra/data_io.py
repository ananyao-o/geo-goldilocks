import pandas as pd
import json
from typing import Dict, Any, List, Tuple

def load_zones(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id","slope","elevation","land_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"zone_features missing columns: {missing}")
    df = df.copy()
    # so we don't run into "Urban" vs "urban" issues in cli
    df["land_type"] = df["land_type"].astype(str).str.strip().str.lower()
    df["point"] = df["id"].apply(lambda x: f"p{x}")
    return df

def load_roads(path: str) -> pd.DataFrame:
    rn = pd.read_csv(path)
    pcols = [c for c in rn.columns if c.lower().startswith("p")]
    if not pcols:
        raise ValueError("road_network.csv must contain p* columns for distances")
    return rn[[*pcols]]  # drop from_point if present

def load_constraints(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def avg_road_distance(roads: pd.DataFrame) -> pd.Series:
    return roads.mean(axis=1)

def build_distance_lookup(roads: pd.DataFrame) -> Dict[Tuple[int,int], float]:
    # roads is NxN distance matrix with columns p1..pN, rows aligned
    pcols = list(roads.columns)
    n = len(pcols)
    lookup = {}
    for i in range(n):
        for j in range(n):
            lookup[(i+1, j+1)] = float(roads.iloc[i, j])
    return lookup
