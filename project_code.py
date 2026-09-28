# NAME: Mohamed Ali | NUMBER: 5182
"""Train a shot-make model and write predictions in the supplied row order.

Run from the repository root: python3 project_code.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "training.csv.gz"
TEST_PATH = ROOT / "testing.csv.gz"
TEMPLATE_PATH = ROOT / "submission.csv"

# Player identifiers have high cardinality. Keep the most common players as
# categories and group the remainder so each categorical field fits the model.
TOP_COUNTS = {"shooter_id": 180, "closestdef_id": 100}
CATEGORICAL = [
    "season_id", "off_team_id", "def_team_id", "shooter_id",
    "closestdef_id", "month", "gamestate", "shottype",
]
NUMERIC = [
    "distance", "locationx", "locationy", "three", "dribblesbefore",
    "shotclock", "closestdefdist", "shooterspeed", "num_contesters",
    "contested", "distcont1", "distcont2", "distcont3", "distcont4",
    "approach_1s", "approach_075s", "approach_05s", "approach_025s",
    "defender_closing_1s", "defender_closing_025s", "min_contester_dist",
]
FEATURES = NUMERIC + CATEGORICAL


def make_features(frame, category_maps=None, top_players=None):
    out = frame[NUMERIC[:14] + CATEGORICAL].copy()
    approach = (
        frame["closestdefapproach"]
        .str.strip("{}")
        .str.split(",", expand=True)
        .apply(pd.to_numeric, errors="coerce")
    )
    approach = approach.reindex(columns=range(4))
    for i, name in enumerate(NUMERIC[14:18]):
        out[name] = approach[i].to_numpy(dtype="float32")
    out["defender_closing_1s"] = out["approach_1s"] - out["closestdefdist"]
    out["defender_closing_025s"] = out["approach_025s"] - out["closestdefdist"]
    out["min_contester_dist"] = frame[
        ["distcont1", "distcont2", "distcont3", "distcont4"]
    ].min(axis=1)

    out["three"] = out["three"].astype("int8")
    out["contested"] = out["contested"].astype("int8")

    if top_players is None:
        top_players = {
            col: set(frame[col].value_counts().head(n).index)
            for col, n in TOP_COUNTS.items()
        }
    if category_maps is None:
        category_maps = {}
        for col in CATEGORICAL:
            series = out[col].fillna("MISSING").astype(str)
            if col in top_players:
                series = series.where(series.isin(top_players[col]), "OTHER")
            category_maps[col] = {v: i for i, v in enumerate(sorted(series.unique()))}

    for col in CATEGORICAL:
        series = out[col].fillna("MISSING").astype(str)
        if col in top_players:
            series = series.where(series.isin(top_players[col]), "OTHER")
        # An unseen category is treated as missing by HistGradientBoosting.
        out[col] = series.map(category_maps[col]).astype("float32")

    out[NUMERIC] = out[NUMERIC].astype("float32")
    return out[FEATURES], category_maps, top_players


def model():
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=200,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        categorical_features=[c in CATEGORICAL for c in FEATURES],
        early_stopping=False,
        random_state=42,
    )


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    template = pd.read_csv(TEMPLATE_PATH)
    if list(template.columns) != ["shot_id", "make_prob"]:
        raise ValueError("Unexpected submission template columns")
    if not template["shot_id"].equals(test["shot_id"]):
        raise ValueError("Submission shot IDs do not match test rows")

    y = train["outcome"].astype("int8")
    fit_idx, valid_idx = train_test_split(
        np.arange(len(train)), test_size=0.20, random_state=42, stratify=y
    )

    # Category rankings for validation are computed using only the fit split.
    x_fit, maps, players = make_features(train.iloc[fit_idx])
    x_valid, _, _ = make_features(train.iloc[valid_idx], maps, players)
    baseline = np.full(len(valid_idx), y.iloc[fit_idx].mean())
    estimator = model()
    estimator.fit(x_fit, y.iloc[fit_idx])
    valid_probs = estimator.predict_proba(x_valid)[:, 1]
    print(f"Constant-probability validation log-loss: {log_loss(y.iloc[valid_idx], baseline):.6f}")
    print(f"Model validation log-loss: {log_loss(y.iloc[valid_idx], valid_probs):.6f}")

    # Model-specific importance: shuffle one validation feature at a time and
    # measure the increase in log-loss. Repeated twice on a fixed 10k sample.
    rng = np.random.default_rng(42)
    sample = rng.choice(len(x_valid), 10_000, replace=False)
    importance_x = x_valid.iloc[sample].copy()
    importance_y = y.iloc[valid_idx].to_numpy()[sample]
    reference_loss = log_loss(importance_y, estimator.predict_proba(importance_x)[:, 1])
    importance = {}
    for col in FEATURES:
        original = importance_x[col].copy()
        changes = []
        for _ in range(2):
            importance_x[col] = rng.permutation(original.to_numpy())
            changes.append(
                log_loss(importance_y, estimator.predict_proba(importance_x)[:, 1])
                - reference_loss
            )
        importance_x[col] = original
        importance[col] = np.mean(changes)
    print("Top permutation importance (increase in validation log-loss):")
    for col, value in sorted(importance.items(), key=lambda item: -item[1])[:10]:
        print(f"  {col:28s} {value:.5f}")

    calibration = pd.DataFrame({"prob": valid_probs, "outcome": y.iloc[valid_idx].to_numpy()})
    calibration["bin"] = pd.cut(calibration["prob"], np.linspace(0, 1, 11), include_lowest=True)
    summary = calibration.groupby("bin", observed=True).agg(
        shots=("outcome", "size"), predicted=("prob", "mean"), actual=("outcome", "mean")
    )
    print("Calibration by predicted-probability bin:")
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))

    # Refit on all labeled shots; recalculate categories from the full train.
    x_train, maps, players = make_features(train)
    x_test, _, _ = make_features(test, maps, players)
    final = model()
    final.fit(x_train, y)
    template["make_prob"] = final.predict_proba(x_test)[:, 1]
    if not template["make_prob"].between(0, 1).all():
        raise ValueError("Invalid probability in output")
    template.to_csv(TEMPLATE_PATH, index=False)
    print(f"Wrote {len(template):,} probabilities to {TEMPLATE_PATH}")


if __name__ == "__main__":
    main()
