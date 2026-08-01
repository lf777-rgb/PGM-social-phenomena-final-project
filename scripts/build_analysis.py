#!/usr/bin/env python3
"""Build the statewide EV charging access PGM analysis."""

from __future__ import annotations

import itertools
import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
(ROOT / "tmp" / "mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
IMG = ROOT / "images"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

STATE_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

REGION_NAMES = {
    1: "Northeast",
    2: "Midwest",
    3: "South",
    4: "West",
}

LEVELS = ["Low", "Middle", "High"]


def load_afdc_ports() -> pd.DataFrame:
    html = (RAW / "afdc_electric_maps_and_data.html").read_text()
    match = re.search(
        r"new window\.ChartModel\((\{\"id\":10366.*?\})\)\);",
        html,
        flags=re.S,
    )
    if not match:
        raise RuntimeError("Could not find AFDC chart 10366 in saved HTML.")
    chart = json.loads(match.group(1))
    ports = pd.DataFrame(chart["data"])
    ports = ports.rename(
        columns={
            "name": "state",
            "public": "public_ports",
            "private": "private_ports",
            "total": "total_ports",
        }
    )
    return ports


def load_ev_registrations() -> pd.DataFrame:
    raw = pd.read_excel(
        RAW / "afdc_ev_registration_counts_by_state_2024.xlsx",
        sheet_name="Condensed",
        header=None,
    )
    reg = raw.iloc[3:, [1, 2]].copy()
    reg.columns = ["state", "ev_registrations_2023"]
    reg = reg.dropna()
    reg["ev_registrations_2023"] = reg["ev_registrations_2023"].astype(int)
    return reg


def load_population() -> pd.DataFrame:
    pop = pd.read_csv(RAW / "census_population_estimates_2023_state.csv")
    states = pop[pop["SUMLEV"].astype(int).eq(40)].copy()
    states = states[states["NAME"].isin(STATE_ABBR)].copy()
    states["region"] = pd.to_numeric(states["REGION"], errors="coerce").astype(int).map(REGION_NAMES)
    return states[["NAME", "region", "POPESTIMATE2023"]].rename(
        columns={"NAME": "state", "POPESTIMATE2023": "population_2023"}
    )


def load_saipe_income_poverty() -> pd.DataFrame:
    rows = []
    for line in (RAW / "census_saipe_2023_us_states.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) < 33:
            continue
        abbreviation = parts[-3]
        state = " ".join(parts[29:-3])
        if abbreviation == "US":
            continue
        rows.append(
            {
                "state": state,
                "all_ages_poverty_count": int(parts[2]),
                "poverty_rate": float(parts[5]),
                "median_household_income": int(parts[20]),
                "saipe_state_abbr": abbreviation,
            }
        )
    return pd.DataFrame(rows)


def load_area() -> pd.DataFrame:
    table = pd.read_html(RAW / "census_tigerweb_state_area_2023.html")[0]
    area = table[table["NAME"].isin(STATE_ABBR)].copy()
    area["land_area_sqmi"] = area["AREALAND"] / 2_589_988.110336
    return area[["NAME", "STUSAB", "land_area_sqmi"]].rename(
        columns={"NAME": "state", "STUSAB": "state_abbr"}
    )


def tertile(series: pd.Series, reverse: bool = False) -> pd.Series:
    labels = list(reversed(LEVELS)) if reverse else LEVELS
    return pd.qcut(series.rank(method="first"), 3, labels=labels).astype(str)


def build_dataset() -> pd.DataFrame:
    df = load_afdc_ports()
    for other in [
        load_ev_registrations(),
        load_population(),
        load_saipe_income_poverty(),
        load_area(),
    ]:
        df = df.merge(other, on="state", how="inner")

    df["public_ports_per_100k"] = df["public_ports"] / df["population_2023"] * 100_000
    df["total_ports_per_100k"] = df["total_ports"] / df["population_2023"] * 100_000
    df["evs_per_1000"] = df["ev_registrations_2023"] / df["population_2023"] * 1_000
    df["population_density"] = df["population_2023"] / df["land_area_sqmi"]
    df["log_population_density"] = np.log1p(df["population_density"])
    df["private_port_share"] = df["private_ports"] / df["total_ports"]
    df["income_level"] = tertile(df["median_household_income"])
    df["poverty_level"] = tertile(df["poverty_rate"])
    df["density_level"] = tertile(df["population_density"])
    df["ev_adoption_level"] = tertile(df["evs_per_1000"])
    df["charging_access_level"] = tertile(df["public_ports_per_100k"])
    return df.sort_values("state").reset_index(drop=True)


def add_context_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = ["median_household_income", "poverty_rate", "log_population_density"]
    scaler = StandardScaler()
    x = scaler.fit_transform(df[features])

    bic_rows = []
    for k in range(2, 6):
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=50, random_state=5650)
        model.fit(x)
        bic_rows.append({"k": k, "bic": model.bic(x)})
    bic = pd.DataFrame(bic_rows)

    model = GaussianMixture(n_components=3, covariance_type="full", n_init=100, random_state=5650)
    comp = model.fit_predict(x)
    work = df.copy()
    work["profile_raw"] = comp

    standardized = pd.DataFrame(x, columns=[f"{c}_z" for c in features])
    standardized["profile_raw"] = comp
    component_scores = standardized.groupby("profile_raw").mean()
    component_scores["advantage_score"] = (
        component_scores["median_household_income_z"]
        - component_scores["poverty_rate_z"]
        + 0.5 * component_scores["log_population_density_z"]
    )
    component_scores["pressure_score"] = (
        -component_scores["median_household_income_z"] + component_scores["poverty_rate_z"]
    )
    affluent_component = component_scores["advantage_score"].idxmax()
    pressured_component = component_scores.drop(index=affluent_component)["pressure_score"].idxmax()
    mixed_component = [idx for idx in component_scores.index if idx not in {affluent_component, pressured_component}][0]
    label_map = {
        pressured_component: "Lower income pressured",
        mixed_component: "Mixed sparse",
        affluent_component: "Affluent dense",
    }
    work["context_profile"] = work["profile_raw"].map(label_map)
    probabilities = pd.DataFrame(
        model.predict_proba(x),
        columns=[label_map[i] for i in range(3)],
    )
    probabilities["state"] = work["state"]
    probabilities["context_profile"] = work["context_profile"]
    probabilities["max_profile_probability"] = probabilities[
        ["Lower income pressured", "Mixed sparse", "Affluent dense"]
    ].max(axis=1)

    profile_summary = (
        work.groupby("context_profile")
        .agg(
            states=("state", "count"),
            median_income=("median_household_income", "mean"),
            poverty_rate=("poverty_rate", "mean"),
            population_density=("population_density", "mean"),
            evs_per_1000=("evs_per_1000", "mean"),
            public_ports_per_100k=("public_ports_per_100k", "mean"),
        )
        .reindex(["Lower income pressured", "Mixed sparse", "Affluent dense"])
        .reset_index()
    )
    return work.drop(columns=["profile_raw"]), bic, profile_summary.merge(
        probabilities.groupby("context_profile")["max_profile_probability"].mean().rename("mean_assignment_probability").reset_index(),
        on="context_profile",
        how="left",
    )


VARIABLES = OrderedDict(
    [
        ("region", ["Northeast", "Midwest", "South", "West"]),
        ("context_profile", ["Lower income pressured", "Mixed sparse", "Affluent dense"]),
        ("income_level", LEVELS),
        ("poverty_level", LEVELS),
        ("density_level", LEVELS),
        ("ev_adoption_level", LEVELS),
        ("charging_access_level", LEVELS),
    ]
)

PARENTS = {
    "region": [],
    "context_profile": ["region"],
    "income_level": ["context_profile"],
    "poverty_level": ["context_profile"],
    "density_level": ["context_profile"],
    "ev_adoption_level": ["context_profile"],
    "charging_access_level": ["context_profile", "ev_adoption_level"],
}


def learn_cpts(data: pd.DataFrame, alpha: float = 1.0) -> dict[str, dict[tuple[str, ...], dict[str, float]]]:
    cpts: dict[str, dict[tuple[str, ...], dict[str, float]]] = {}
    for var, levels in VARIABLES.items():
        parents = PARENTS[var]
        cpts[var] = {}
        if not parents:
            counts = data[var].value_counts().reindex(levels, fill_value=0).astype(float) + alpha
            probs = counts / counts.sum()
            cpts[var][()] = probs.to_dict()
            continue
        parent_levels = [VARIABLES[p] for p in parents]
        for parent_values in itertools.product(*parent_levels):
            subset = data
            for p, val in zip(parents, parent_values):
                subset = subset[subset[p].eq(val)]
            counts = subset[var].value_counts().reindex(levels, fill_value=0).astype(float) + alpha
            probs = counts / counts.sum()
            cpts[var][tuple(parent_values)] = probs.to_dict()
    return cpts


def row_probability(row: pd.Series, cpts: dict[str, dict[tuple[str, ...], dict[str, float]]]) -> float:
    prob = 1.0
    for var in VARIABLES:
        key = tuple(row[p] for p in PARENTS[var])
        prob *= cpts[var][key][row[var]]
    return prob


def enumerate_query(
    cpts: dict[str, dict[tuple[str, ...], dict[str, float]]],
    target: str,
    evidence: dict[str, str] | None = None,
) -> dict[str, float]:
    evidence = evidence or {}
    totals = {level: 0.0 for level in VARIABLES[target]}
    denominator = 0.0
    keys = list(VARIABLES.keys())
    for values in itertools.product(*[VARIABLES[k] for k in keys]):
        assignment = dict(zip(keys, values))
        if any(assignment[k] != v for k, v in evidence.items()):
            continue
        p = 1.0
        for var in keys:
            parent_key = tuple(assignment[pv] for pv in PARENTS[var])
            p *= cpts[var][parent_key][assignment[var]]
        totals[assignment[target]] += p
        denominator += p
    if denominator == 0:
        return {k: math.nan for k in totals}
    return {k: v / denominator for k, v in totals.items()}


def leave_one_out_eval(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    eps = 1e-15
    for i in range(len(df)):
        train = df.drop(index=i)
        test = df.iloc[i]
        cpts = learn_cpts(train)
        posterior = enumerate_query(
            cpts,
            target="charging_access_level",
            evidence={
                "context_profile": test["context_profile"],
                "ev_adoption_level": test["ev_adoption_level"],
            },
        )
        baseline = enumerate_query(cpts, target="charging_access_level")
        pred = max(posterior, key=posterior.get)
        baseline_pred = max(baseline, key=baseline.get)
        rows.append(
            {
                "state": test["state"],
                "observed": test["charging_access_level"],
                "predicted": pred,
                "p_low": posterior["Low"],
                "p_middle": posterior["Middle"],
                "p_high": posterior["High"],
                "log_loss": -math.log(max(posterior[test["charging_access_level"]], eps)),
                "baseline_predicted": baseline_pred,
                "baseline_log_loss": -math.log(max(baseline[test["charging_access_level"]], eps)),
            }
        )
    pred_df = pd.DataFrame(rows)
    metrics = {
        "loo_accuracy": float((pred_df["observed"] == pred_df["predicted"]).mean()),
        "chance_accuracy": float(1 / len(LEVELS)),
        "loo_log_loss": float(pred_df["log_loss"].mean()),
        "baseline_log_loss": float(pred_df["baseline_log_loss"].mean()),
    }
    return pred_df, metrics


def cpt_to_frame(cpts: dict[str, dict[tuple[str, ...], dict[str, float]]], var: str) -> pd.DataFrame:
    rows = []
    parents = PARENTS[var]
    for parent_values, probs in cpts[var].items():
        base = {p: v for p, v in zip(parents, parent_values)}
        for level, prob in probs.items():
            rows.append({**base, var: level, "probability": prob})
    return pd.DataFrame(rows)


def serialize_cpts(cpts: dict[str, dict[tuple[str, ...], dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    serial = {}
    for var, table in cpts.items():
        serial[var] = {" | ".join(parent_values) if parent_values else "(root)": probs for parent_values, probs in table.items()}
    return serial


def model_log_likelihood(df: pd.DataFrame, cpts: dict[str, dict[tuple[str, ...], dict[str, float]]]) -> float:
    return float(np.log([row_probability(row, cpts) for _, row in df.iterrows()]).sum())


def model_parameter_count() -> int:
    count = 0
    for var, levels in VARIABLES.items():
        parent_configs = 1
        for p in PARENTS[var]:
            parent_configs *= len(VARIABLES[p])
        count += parent_configs * (len(levels) - 1)
    return count


def plot_dag() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    positions = {
        "Region": (0.10, 0.78),
        "Context profile": (0.38, 0.78),
        "Income": (0.15, 0.47),
        "Poverty": (0.36, 0.47),
        "Density": (0.57, 0.47),
        "EV adoption": (0.73, 0.78),
        "Charging access": (0.72, 0.23),
    }
    edges = [
        ("Region", "Context profile"),
        ("Context profile", "Income"),
        ("Context profile", "Poverty"),
        ("Context profile", "Density"),
        ("Context profile", "EV adoption"),
        ("Context profile", "Charging access"),
        ("EV adoption", "Charging access"),
    ]
    for name, (x, y) in positions.items():
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#F7F4EA", edgecolor="#273043", linewidth=1.4),
        )
    for src, dst in edges:
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        ax.annotate(
            "",
            xy=(x2, y2 + 0.035 if y2 < y1 else y2 - 0.04),
            xytext=(x1, y1 - 0.04 if y2 < y1 else y1 + 0.04),
            arrowprops=dict(arrowstyle="->", color="#273043", lw=1.5, shrinkA=8, shrinkB=8),
        )
    ax.set_title("Bayesian network used for state EV charging access", fontsize=14, pad=20)
    fig.tight_layout()
    fig.savefig(IMG / "pgm_dag.svg", bbox_inches="tight")
    plt.close(fig)


def plot_figures(df: pd.DataFrame, profile_summary: pd.DataFrame, charging_cpt: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    palette = {
        "Lower income pressured": "#B85750",
        "Mixed sparse": "#4C7A92",
        "Affluent dense": "#3B8D64",
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    sns.scatterplot(
        data=df,
        x="evs_per_1000",
        y="public_ports_per_100k",
        hue="context_profile",
        size="population_2023",
        sizes=(40, 550),
        palette=palette,
        alpha=0.82,
        ax=ax,
    )
    for _, row in df.nlargest(4, "public_ports_per_100k").iterrows():
        ax.text(row["evs_per_1000"] + 0.5, row["public_ports_per_100k"], row["state_abbr"], fontsize=9)
    for _, row in df.nlargest(3, "evs_per_1000").iterrows():
        ax.text(row["evs_per_1000"] + 0.5, row["public_ports_per_100k"], row["state_abbr"], fontsize=9)
    ax.set_xlabel("EV registrations per 1,000 residents")
    ax.set_ylabel("Public charging ports per 100,000 residents")
    ax.set_title("EV adoption and public charging access by latent context profile")
    ax.legend(loc="best", frameon=True, title="")
    fig.tight_layout()
    fig.savefig(IMG / "ev_vs_ports.svg", bbox_inches="tight")
    plt.close(fig)

    scaled = profile_summary.copy()
    metrics = [
        ("median_income", "Median income"),
        ("poverty_rate", "Poverty rate"),
        ("population_density", "Population density"),
        ("evs_per_1000", "EVs per 1,000"),
        ("public_ports_per_100k", "Ports per 100,000"),
    ]
    rows = []
    for col, label in metrics:
        vals = scaled[col]
        z = (vals - vals.mean()) / vals.std(ddof=0)
        for profile, value in zip(scaled["context_profile"], z):
            rows.append({"context_profile": profile, "metric": label, "standardized_mean": value})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        data=pd.DataFrame(rows),
        x="metric",
        y="standardized_mean",
        hue="context_profile",
        palette=palette,
        ax=ax,
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Standardized profile mean")
    ax.set_title("Latent profiles summarize social and infrastructure differences")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(IMG / "profile_summary.svg", bbox_inches="tight")
    plt.close(fig)

    heat = (
        charging_cpt[charging_cpt["charging_access_level"].eq("High")]
        .pivot(index="context_profile", columns="ev_adoption_level", values="probability")
        .reindex(index=["Lower income pressured", "Mixed sparse", "Affluent dense"], columns=LEVELS)
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={"label": "P(high charging access)"}, ax=ax)
    ax.set_xlabel("EV adoption level")
    ax.set_ylabel("Latent context profile")
    ax.set_title("Conditional probability of high charging access")
    fig.tight_layout()
    fig.savefig(IMG / "charging_cpt_heatmap.svg", bbox_inches="tight")
    plt.close(fig)

    plot_dag()


def main() -> None:
    df = build_dataset()
    df, bic, profile_summary = add_context_profiles(df)
    cpts = learn_cpts(df)
    charging_cpt = cpt_to_frame(cpts, "charging_access_level")
    predictions, eval_metrics = leave_one_out_eval(df)

    correlations = []
    pairs = [
        ("median_household_income", "public_ports_per_100k"),
        ("poverty_rate", "public_ports_per_100k"),
        ("population_density", "public_ports_per_100k"),
        ("evs_per_1000", "public_ports_per_100k"),
    ]
    for x, y in pairs:
        rho, p = spearmanr(df[x], df[y])
        correlations.append({"x": x, "y": y, "spearman_rho": float(rho), "p_value": float(p)})
    correlations = pd.DataFrame(correlations)

    ll = model_log_likelihood(df, cpts)
    params = model_parameter_count()
    n = len(df)
    metrics = {
        **eval_metrics,
        "states": int(n),
        "model_log_likelihood": ll,
        "model_parameters": int(params),
        "model_bic": float(-2 * ll + params * math.log(n)),
        "median_ports_per_100k": float(df["public_ports_per_100k"].median()),
        "median_evs_per_1000": float(df["evs_per_1000"].median()),
        "mean_profile_assignment_probability": float(
            np.average(profile_summary["mean_assignment_probability"], weights=profile_summary["states"])
        ),
        "p_high_charging_given_high_ev": enumerate_query(
            cpts, "charging_access_level", {"ev_adoption_level": "High"}
        )["High"],
        "p_high_charging_given_low_ev": enumerate_query(
            cpts, "charging_access_level", {"ev_adoption_level": "Low"}
        )["High"],
        "p_high_charging_affluent_high_ev": enumerate_query(
            cpts,
            "charging_access_level",
            {"context_profile": "Affluent dense", "ev_adoption_level": "High"},
        )["High"],
        "p_high_charging_lower_low_ev": enumerate_query(
            cpts,
            "charging_access_level",
            {"context_profile": "Lower income pressured", "ev_adoption_level": "Low"},
        )["High"],
    }

    top_ports = df.sort_values("public_ports_per_100k", ascending=False).head(10)
    bottom_ports = df.sort_values("public_ports_per_100k", ascending=True).head(10)
    profile_states = (
        df.groupby("context_profile")["state_abbr"]
        .apply(lambda x: ", ".join(sorted(x)))
        .reindex(["Lower income pressured", "Mixed sparse", "Affluent dense"])
        .reset_index(name="states")
    )

    columns = [
        "state",
        "state_abbr",
        "region",
        "population_2023",
        "land_area_sqmi",
        "median_household_income",
        "poverty_rate",
        "population_density",
        "ev_registrations_2023",
        "public_ports",
        "private_ports",
        "total_ports",
        "evs_per_1000",
        "public_ports_per_100k",
        "income_level",
        "poverty_level",
        "density_level",
        "ev_adoption_level",
        "charging_access_level",
        "context_profile",
    ]
    df[columns].to_csv(OUT / "state_model_data.csv", index=False)
    profile_summary.to_csv(OUT / "profile_summary.csv", index=False)
    profile_states.to_csv(OUT / "profile_states.csv", index=False)
    bic.to_csv(OUT / "gmm_bic_table.csv", index=False)
    charging_cpt.to_csv(OUT / "cpt_charging_access.csv", index=False)
    predictions.to_csv(OUT / "loo_predictions.csv", index=False)
    correlations.to_csv(OUT / "correlations.csv", index=False)
    top_ports[["state", "public_ports_per_100k", "evs_per_1000", "context_profile"]].to_csv(
        OUT / "top_public_ports_per_capita.csv", index=False
    )
    bottom_ports[["state", "public_ports_per_100k", "evs_per_1000", "context_profile"]].to_csv(
        OUT / "bottom_public_ports_per_capita.csv", index=False
    )
    (OUT / "summary_metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "model_cpts.json").write_text(json.dumps(serialize_cpts(cpts), indent=2))

    plot_figures(df, profile_summary, charging_cpt)

    print(json.dumps(metrics, indent=2))
    print("\nProfile summary:")
    print(profile_summary.to_string(index=False))
    print("\nTop states by public ports per 100k residents:")
    print(top_ports[["state", "public_ports_per_100k", "evs_per_1000", "context_profile"]].to_string(index=False))


if __name__ == "__main__":
    main()
