from __future__ import annotations
import pandas as pd
from typing import Dict, Tuple, List
import pulp

def build_conflicts(distance_lookup: Dict[Tuple[int,int], float], min_distance: float, n: int) -> List[Tuple[int,int]]:
    """
    build pairs (i,j) where both cannot be selected since dist(i,j) < min_distance.
    """
    conflicts = []
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            d = distance_lookup[(i,j)]
            if d < float(min_distance):
                conflicts.append((i,j))
    return conflicts

def solve_milp(scores: pd.Series, feasible: pd.Series, k: int, conflicts: List[Tuple[int,int]]):
    """
    minimise sum(scores[i] * x_i) s.t. sum x_i = k, conflicts, and x_i <= feasible_i.
    """
    idx = list(scores.index)
    prob = pulp.LpProblem("zone_selection", pulp.LpMinimize)
    x = {i: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary) for i in idx}

    prob += pulp.lpSum([float(scores[i]) * x[i] for i in idx])
    prob += pulp.lpSum([x[i] for i in idx]) == k # to get exactly k selections

    # feasibility constraints
    for i in idx:
        if not bool(feasible[i]):
            prob += x[i] == 0

    # conflict constraints
    for (i,j) in conflicts:
        if i in x and j in x:
            prob += x[i] + x[j] <= 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    chosen = [i for i in idx if pulp.value(x[i]) >= 0.5]
    obj = pulp.value(prob.objective)
    return chosen, obj, status
