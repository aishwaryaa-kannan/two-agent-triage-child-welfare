"""
Validation-based protocol for the two-agent triage framework.

Per seed (fresh synthetic data, split and model seed):
  - 25% test split (never used for any choice)
  - remaining 75% split 80/20 into train / validation
  - class weight in {None, balanced} and margin in {0.05, 0.10, 0.15, 0.20} are chosen by
    minimizing a validation cost:  cost = c_FN * (unflagged FN / n) + (flag rate)
    i.e. one unit per human review, c_FN units per unstable case that is neither flagged nor correct.
    Human reviewers are assumed to decide flagged cases correctly (a strong assumption).
  - the chosen configuration is refit on train+validation and evaluated once on test.
Run with c_FN = 5 and c_FN = 10; both are reported (neither was singled out after seeing results).
"""
from triage_lib import *

CWS = [None, "balanced"]
MARGINS = [0.05, 0.10, 0.15, 0.20]
CFNS = [5, 10]
SEEDS = list(range(20))


def fit_predict(Xtr, ytr, Xev, cw, seed):
    A = make_pipe(CAT_A, RandomForestClassifier(n_estimators=200, random_state=seed, class_weight=cw))
    B = make_pipe(CAT_B, LogisticRegression(max_iter=1000, class_weight=cw))
    A.fit(Xtr[FEAT_A], ytr)
    B.fit(Xtr[FEAT_B], ytr)
    return A.predict_proba(Xev[FEAT_A])[:, 1], B.predict_proba(Xev[FEAT_B])[:, 1]


def flag_full(pA, pB, margin):
    dis = (pA >= 0.5) != (pB >= 0.5)
    return dis | (np.abs(pA - 0.5) < margin) | (np.abs(pB - 0.5) < margin)


def cost(y, predA, flag, c_fn):
    fn = ((y == 1) & (predA == 0) & ~flag).sum()
    return c_fn * fn / len(y) + flag.mean()


rows = []
for seed in SEEDS:
    df = gen(seed)
    y = df["prolonged_unstable_stay"]
    X = df.drop(columns=["prolonged_unstable_stay"])
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.20, random_state=seed, stratify=ytv)
    yva_v, yte_v = yva.values, yte.values

    val = {}
    for cw in CWS:
        pA, pB = fit_predict(Xtr, ytr, Xva, cw, seed)
        predA = (pA >= 0.5).astype(int)
        for m in MARGINS:
            f = flag_full(pA, pB, m)
            for c in CFNS:
                val[(cw, m, c)] = cost(yva_v, predA, f, c)

    test_p = {cw: fit_predict(Xtv, ytv, Xte, cw, seed) for cw in CWS}
    rng = np.random.default_rng(seed)
    n = len(yte_v)
    for c in CFNS:
        cw, m = min(((cw, m) for cw in CWS for m in MARGINS), key=lambda t: val[(t[0], t[1], c)])
        pA, pB = test_p[cw]
        predA = (pA >= 0.5).astype(int)
        flag = flag_full(pA, pB, m)
        keep = ~flag
        k = int(keep.sum())
        mt = sel_metrics(yte_v, predA, keep)
        base = sel_metrics(yte_v, predA, topk_mask(np.abs(pA - 0.5), k, rng))
        rnd = random_baseline(yte_v, predA, k, rng, R=50)
        allm = sel_metrics(yte_v, predA, np.ones(n, bool))
        flag_rate = 1 - mt["cov"]
        rows.append(dict(
            c_fn=c, seed=seed, cw=str(cw), margin=m, cov=mt["cov"], flag_rate=flag_rate,
            acc=mt["acc"], f1=mt["f1"], rec=mt["rec"], fn_per_case=mt["fn"] / n,
            Amargin_acc=base["acc"], Amargin_fn_per_case=base["fn"] / n,
            random_acc=rnd["acc"], random_fn_per_case=rnd["fn"] / n,
            A_all_acc=allm["acc"], A_all_f1=allm["f1"], A_all_fn_per_case=allm["fn"] / n,
            cost_triage=c * mt["fn"] / n + flag_rate, cost_A_alone=c * allm["fn"] / n, cost_flag_all=1.0,
            breakeven_cfn=flag_rate / max((allm["fn"] - mt["fn"]) / n, 1e-9)))

res = pd.DataFrame(rows)
res.to_csv("results_protocol.csv", index=False)

COLS = ["cov", "flag_rate", "acc", "f1", "rec", "fn_per_case", "Amargin_acc", "Amargin_fn_per_case",
        "random_acc", "random_fn_per_case", "A_all_acc", "A_all_fn_per_case",
        "cost_triage", "cost_A_alone", "breakeven_cfn"]
for c in CFNS:
    d = res[res.c_fn == c]
    print("=" * 78)
    print(f"c_FN = {c}   ({len(d)} seeds, test-set results of the validation-selected configuration)")
    print("=" * 78)
    print("selected (class_weight, margin), counts over seeds:")
    print(d.groupby(["cw", "margin"]).size().to_string())
    print(pd.DataFrame({"mean": d[COLS].mean(), "std": d[COLS].std()}).round(3).to_string())
    dacc = d.acc - d.Amargin_acc
    dfn = d.fn_per_case - d.Amargin_fn_per_case
    print(f"vs A-confidence-only at same coverage: acc diff {dacc.mean():+.4f} "
          f"(better in {(dacc > 0).sum()}/{len(d)}); FN-rate diff {dfn.mean():+.4f} "
          f"(lower in {(dfn < 0).sum()}/{len(d)})")
    print(f"cost: triage < A alone in {(d.cost_triage < d.cost_A_alone).sum()}/{len(d)} seeds; "
          f"triage < flag-all (1.0) in {(d.cost_triage < 1.0).sum()}/{len(d)} seeds")
print("\nsaved results_protocol.csv")
