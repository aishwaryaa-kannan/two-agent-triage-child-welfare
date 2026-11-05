"""
Full evaluation for the two-agent triage paper (single script, reproduces all numbers).

Sections
  1. Seed-42 reproduction of the paper's numbers (unweighted and class-balanced)
  2. Ablation of the flagging signals, each compared with an A-margin baseline at the SAME coverage
  3. Multi-seed evaluation (new synthetic data + split + model seed per run)
  4. Risk-coverage curve (saved as risk_coverage.png / .pdf)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score

FEAT_A = ["age_at_removal", "prior_placements", "removal_reason",
          "has_disability", "sibling_group_size", "case_goal"]
FEAT_B = ["prior_placements", "has_disability", "sibling_group_size", "removal_reason"]
CAT_A = ["removal_reason", "case_goal"]
CAT_B = ["removal_reason"]
MARGIN = 0.1


def gen(seed, N=3000):
    rng = np.random.default_rng(seed)
    age = rng.integers(0, 18, N)
    prior = rng.poisson(1.2, N)
    reason = rng.choice(["neglect", "abuse", "parental_incapacity", "abandonment", "other"],
                        N, p=[0.45, 0.20, 0.20, 0.10, 0.05])
    dis = rng.choice([0, 1], N, p=[0.85, 0.15])
    sib = rng.choice([1, 2, 3, 4], N, p=[0.55, 0.25, 0.12, 0.08])
    goal = rng.choice(["reunification", "adoption", "guardianship", "long_term_care"],
                      N, p=[0.5, 0.25, 0.15, 0.10])
    risk = (0.15 * prior + 0.10 * (age > 12).astype(int) + 0.20 * dis
            + 0.10 * (sib > 2).astype(int) + 0.15 * (reason == "abuse").astype(int)
            + 0.10 * (goal == "long_term_care").astype(int) + rng.normal(0, 0.3, N))
    y = (risk > np.quantile(risk, 0.65)).astype(int)
    return pd.DataFrame({"age_at_removal": age, "prior_placements": prior,
                         "removal_reason": reason, "has_disability": dis,
                         "sibling_group_size": sib, "case_goal": goal,
                         "prolonged_unstable_stay": y})


def make_pipe(cat, model):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
                            remainder="passthrough")
    return Pipeline([("pre", pre), ("clf", model)])


def run_agents(data_seed, split_seed, cw):
    df = gen(data_seed)
    y = df["prolonged_unstable_stay"]
    X = df.drop(columns=["prolonged_unstable_stay"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=split_seed, stratify=y)
    A = make_pipe(CAT_A, RandomForestClassifier(n_estimators=200, random_state=split_seed, class_weight=cw))
    B = make_pipe(CAT_B, LogisticRegression(max_iter=1000, class_weight=cw))
    A.fit(Xtr[FEAT_A], ytr)
    B.fit(Xtr[FEAT_B], ytr)
    pA = A.predict_proba(Xte[FEAT_A])[:, 1]
    pB = B.predict_proba(Xte[FEAT_B])[:, 1]
    return yte.values, pA, pB


def sel_metrics(y, pred, keep):
    yk, pk = y[keep], pred[keep]
    return dict(cov=keep.mean(),
                acc=accuracy_score(yk, pk),
                f1=f1_score(yk, pk, zero_division=0),
                rec=recall_score(yk, pk, zero_division=0),
                fn=int(((yk == 1) & (pk == 0)).sum()))


def topk_mask(score, k, rng):
    s = score + 1e-9 * rng.random(len(score))
    idx = np.argsort(-s)[:k]
    m = np.zeros(len(score), bool)
    m[idx] = True
    return m


def random_baseline(y, pred, k, rng, R=100):
    out = []
    for _ in range(R):
        m = np.zeros(len(y), bool)
        m[rng.choice(len(y), k, replace=False)] = True
        out.append(sel_metrics(y, pred, m))
    return {key: float(np.mean([o[key] for o in out])) for key in out[0]}


def flag_rules(pA, pB):
    predA, predB = (pA >= 0.5).astype(int), (pB >= 0.5).astype(int)
    dis = predA != predB
    mA, mB = np.abs(pA - 0.5), np.abs(pB - 0.5)
    return predA, dis, mA, mB, {
        "Disagreement only": dis,
        "Margin A only (Chow)": mA < MARGIN,
        "Margin A or B": (mA < MARGIN) | (mB < MARGIN),
        "Full (disagree or margin)": dis | (mA < MARGIN) | (mB < MARGIN),
    }


def fmt(df, nd=3):
    return df.round(nd).to_string()
