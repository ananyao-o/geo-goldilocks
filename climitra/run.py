import argparse, json, os, datetime
import pandas as pd
from .data_io import load_zones, load_roads, load_constraints, avg_road_distance, build_distance_lookup
from .ml_model import make_features, train_model, predict_scores
from .optimizer import build_conflicts, solve_milp
import matplotlib
matplotlib.use("Agg")  # ensure saving works in headless/CLI runs
import matplotlib.pyplot as plt



def parse_args():
    p = argparse.ArgumentParser(description="Climitra ML+MILP Ranker")
    p.add_argument('--resource_points', type=str, default=None, 
                   help='Optional CSV with columns: id, latitude/longitude (or lat/long/lng)')
    p.add_argument('--zones', required=True, help='Path to zone_features.csv')
    p.add_argument('--roads', required=True, help='Path to road_network.csv')
    p.add_argument('--constraints', required=True, help='Path to constraints.json')
    p.add_argument('--k', type=int, default=3, help='Top-K zones to select (default: 3)')
    # for cli overrides
    p.add_argument('--max_slope', type=float, help='Override max_slope')
    p.add_argument('--exclude_land_types', type=str, help='Comma-separated land types to exclude')
    p.add_argument('--min_distance_km', type=float, help='Override min_distance_from_each_other_km')
    p.add_argument('--outdir', type=str, default=None, help='Output directory (default: outputs/timestamp)')
    return p.parse_args()

def apply_overrides(base: dict, args) -> dict:
    c = dict(base)
    if args.max_slope is not None:
        c["max_slope"] = float(args.max_slope)
    if args.exclude_land_types is not None:
        c["exclude_land_types"] = [s.strip() for s in args.exclude_land_types.split(",") if s.strip()]
    if args.min_distance_km is not None:
        c["min_distance_from_each_other_km"] = float(args.min_distance_km)
    return c

def main():
    args = parse_args()

    zones = load_zones(args.zones)
    roads = load_roads(args.roads)
    constraints = load_constraints(args.constraints)
    constraints = apply_overrides(constraints, args)

    # build our base features (uses roads for avg distance)
    avg = avg_road_distance(roads)

    # now compute feasibility from constraints
    max_slope = constraints.get("max_slope", None)
    excl = set([str(s).strip().lower() for s in constraints.get("exclude_land_types", [])])
    feas_series = (zones["land_type"].apply(lambda x: str(x).strip().lower() not in excl))
    if max_slope is not None:
        feas_series &= (zones["slope"] <= float(max_slope))
    feas_series = feas_series.astype(bool)

    # train the ML model on feasible subset only to enforce constraints
    feat_all = make_features(zones, avg)
    feat_train = feat_all[feas_series.values].reset_index(drop=True)
    if len(feat_train) == 0:
        raise RuntimeError("No feasible zones under current constraints. Relax constraints and try again.")

    model = train_model(feat_train)

    # we predict on all to better inspect scores, but we'll mask infeasible later
    scores_all = pd.Series(predict_scores(model, feat_all), index=zones["id"])

    # MILP selection with spacing conflicts (feasible only) ---
    n = len(zones)
    dist_lookup = build_distance_lookup(roads)
    min_d = float(constraints.get("min_distance_from_each_other_km", 0.0))
    conflicts = build_conflicts(dist_lookup, min_d, n)

    # now mask infeasible by giving them NaN and disallow selection
    scores = scores_all.copy()
    feasible_mask = pd.Series(feas_series.values, index=zones["id"])
    scores[~feasible_mask] = float("nan")

    # if feasible count < k, reduce k and note it
    feasible_count = int(feasible_mask.sum())
    k_eff = min(args.k, feasible_count)
    if k_eff < args.k:
        print(f"[WARN] Only {feasible_count} feasible zones; reducing K from {args.k} to {k_eff}.")

    chosen, obj, status = solve_milp(scores.dropna(), feasible_mask.loc[scores.dropna().index], k_eff, conflicts)

    # build ranking list (feasible only) in ascending order
    df = zones.copy()
    df["avg_road_distance"] = avg.values
    df["ml_score"] = df["id"].map(scores_all)  # raw score (before masking), for transparency
    df["feasible"] = df["id"].map(feasible_mask)
    ranked = df[df["feasible"]].copy()
    ranked = ranked.sort_values("ml_score").reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    # to save outputs
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", ts)
    os.makedirs(outdir, exist_ok=True)

    ranking_csv = os.path.join(outdir, "ranking.csv")
    ranked[["rank","id","slope","elevation","land_type","avg_road_distance","ml_score"]].to_csv(ranking_csv, index=False)

    topk_json = os.path.join(outdir, "top_k.json")
    topk = ranked[ranked["id"].isin(chosen)].sort_values("ml_score")
    with open(topk_json, "w", encoding="utf-8") as f:
        json.dump({
            "k": k_eff,
            "selected_ids": list(topk["id"]),
            "zones": topk.to_dict(orient="records"),
            "objective": obj,
            "solver_status": int(status)
        }, f, indent=2)

    # to save justification file
    just_path = os.path.join(outdir, "justification.txt")
    total = len(zones)
    feasible_cnt = int(feasible_mask.sum())
    infeasible_cnt = total - feasible_cnt
    excl_types = ", ".join(sorted(excl)) if excl else "None"

    text = (
        f"Climitra - Run Justification\n"
        f"- Total zones: {total}\n"
        f"- Feasible after constraints: {feasible_cnt} (Infeasible: {infeasible_cnt})\n"
        f"- Constraints used (applied to training & selection): "
        f"max_slope={max_slope}, exclude_land_types=[{excl_types}], "
        f"min_distance_from_each_other_km={min_d}\n"
        f"- Conflict pairs (distance < {min_d} km): {len(conflicts)}\n"
        "- Effect: The ML model is retrained on the feasible subset only, so changing constraints shifts the training "
        "distribution and score landscape. The MILP then optimises over the feasible pool with spacing conflicts, "
        f"which can change the final Top {args.k}.\n"
    )

    with open(just_path, "w", encoding="utf-8") as f:
        f.write(text)
        #######################################################################
            # Load optional resource points with lat/lon
    rp = None
    if args.resource_points:
        rp = pd.read_csv(args.resource_points)
        # normalise headers
        rp = rp.rename(columns=lambda c: str(c).strip().lower())
        # map aliases
        alias_map = {"lat": "latitude", "long": "longitude", "lng": "longitude"}
        for old, new in alias_map.items():
            if old in rp.columns and new not in rp.columns:
                rp = rp.rename(columns={old: new})
        # must have id + latitude + longitude
        if not {"id", "latitude", "longitude"}.issubset(rp.columns):
            raise RuntimeError(
                "resource_points must have columns: id, latitude, longitude (aliases lat/long/lng allowed). "
                f"Found: {list(rp.columns)}"
            )
        # keep only what we need
        rp = rp[["id", "latitude", "longitude"]].copy()

                                   # --- Simple Lat/Lon scatter plot taken from resource points when provided ---
    # Build a plotting frame: start from zones and add feasible/selected flags
    plot_df = zones.copy()
    plot_df["ml_score"] = plot_df["id"].map(scores_all)
    plot_df["feasible"] = plot_df["id"].map(feasible_mask)
    plot_df["selected"] = plot_df["id"].isin(set(chosen))

    # If resource points are provided, merge lat/lon by id; else expect lat/lon in zones
    if rp is not None:
        plot_df = plot_df.merge(rp, on="id", how="left")
    else:
        # normalise headers for zones in case they already have lat/lon
        plot_df = plot_df.rename(columns=lambda c: str(c).strip().lower())
        alias_map = {"lat": "latitude", "long": "longitude", "lng": "longitude"}
        for old, new in alias_map.items():
            if old in plot_df.columns and new not in plot_df.columns:
                plot_df = plot_df.rename(columns={old: new})

    if not {"latitude", "longitude"}.issubset(plot_df.columns):
        raise RuntimeError(
            "No latitude/longitude found. Provide --resource_points CSV with id, latitude, longitude "
            "or ensure zones has latitude/longitude."
        )

    # Plot layers
    infeas = plot_df[~plot_df["feasible"] & plot_df["longitude"].notna() & plot_df["latitude"].notna()]
    feas   = plot_df[plot_df["feasible"] & ~plot_df["selected"] &
                     plot_df["longitude"].notna() & plot_df["latitude"].notna()]
    picked = plot_df[plot_df["selected"] & plot_df["longitude"].notna() & plot_df["latitude"].notna()]

    plt.figure(figsize=(8, 6))
    if not infeas.empty:
        plt.scatter(infeas["longitude"], infeas["latitude"], c="lightgrey", s=15, label="Infeasible", alpha=0.85)
    if not feas.empty:
        plt.scatter(feas["longitude"], feas["latitude"], c="red", s=20, label="Feasible", alpha=0.85)
    if not picked.empty:
        # draw selected last so stars are on top
        plt.scatter(picked["longitude"], picked["latitude"], marker="*", c="yellow", s=180,
                    edgecolors="black", linewidths=0.8, label=f"Selected (Top-{k_eff})")
        

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Zone Feasibility & Top-K Selection (Lat vs Lon)")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    plot_path = os.path.join(outdir, "selection_plot.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved plot to:    {plot_path}")
    ###########################################################






    print(f"Saved ranking to: {ranking_csv}")
    print(f"Saved top-K to:   {topk_json}")
    print(f"Saved notes to:   {just_path}")


if __name__ == '__main__':
    main()
