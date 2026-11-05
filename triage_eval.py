from triage_lib import *

# ---------------------------------------------------------------- 1. seed-42 reproduction
print("=" * 78)
print("1. SEED-42 REPRODUCTION (matches the paper's setup)")
print("=" * 78)
rows = []
for cw, label in [(None, "unweighted"), ("balanced", "balanced")]:
    y, pA, pB = run_agents(42, 42, cw)
    predA, dis, mA, mB, rules = flag_rules(pA, pB)
    allm = sel_metrics(y, predA, np.ones(len(y), bool))
    rows.append(dict(system=f"A alone, all cases ({label})", **allm))
    keep = ~rules["Full (disagree or margin)"]
    rows.append(dict(system=f"Two-agent, confident subset ({label})", **sel_metrics(y, predA, keep)))
t1 = pd.DataFrame(rows).set_index("system")
t1["flag_rate"] = 1 - t1["cov"]
print(fmt(t1[["cov", "flag_rate", "acc", "f1", "rec", "fn"]]))
t1.to_csv("results_seed42.csv")

# ---------------------------------------------------------------- 2. ablation vs matched-coverage baselines
print("\n" + "=" * 78)
print("2. ABLATION (seed 42, balanced). Each rule vs A-margin top-k and random at SAME coverage")
print("=" * 78)
y, pA, pB = run_agents(42, 42, "balanced")
predA, dis, mA, mB, rules = flag_rules(pA, pB)
rng = np.random.default_rng(0)
rows = []
for name, flag in rules.items():
    keep = ~flag
    k = int(keep.sum())
    m = sel_metrics(y, predA, keep)
    base = sel_metrics(y, predA, topk_mask(mA, k, rng))
    rnd = random_baseline(y, predA, k, rng)
    rows.append(dict(rule=name, cov=m["cov"], acc=m["acc"], f1=m["f1"], fn=m["fn"],
                     Amargin_acc=base["acc"], Amargin_f1=base["f1"], Amargin_fn=base["fn"],
                     random_acc=rnd["acc"], random_fn=rnd["fn"]))
abl = pd.DataFrame(rows).set_index("rule")
print(fmt(abl))
abl.to_csv("results_ablation.csv")

# ---------------------------------------------------------------- 3. multi-seed
print("\n" + "=" * 78)
print("3. MULTI-SEED (20 runs; fresh synthetic data, split and model seed each run)")
print("=" * 78)
SEEDS = list(range(20))
GRID = np.linspace(0.1, 1.0, 19)
ms_rows, rc = [], {}
for cw, label in [(None, "unweighted"), ("balanced", "balanced")]:
    rc[label] = {s: [] for s in ("Full two-agent", "Margin A or B", "Margin A only", "Random")}
    for seed in SEEDS:
        y, pA, pB = run_agents(seed, seed, cw)
        predA, dis, mA, mB, rules = flag_rules(pA, pB)
        rng = np.random.default_rng(seed)
        keep = ~rules["Full (disagree or margin)"]
        k = int(keep.sum())
        m = sel_metrics(y, predA, keep)
        base = sel_metrics(y, predA, topk_mask(mA, k, rng))
        rnd = random_baseline(y, predA, k, rng, R=50)
        allm = sel_metrics(y, predA, np.ones(len(y), bool))
        ms_rows.append(dict(setting=label, seed=seed, cov=m["cov"], full_acc=m["acc"], full_f1=m["f1"],
                            full_fn=m["fn"], Amargin_acc=base["acc"], Amargin_f1=base["f1"],
                            Amargin_fn=base["fn"], random_acc=rnd["acc"], random_fn=rnd["fn"],
                            A_all_acc=allm["acc"], A_all_f1=allm["f1"]))
        # risk-coverage curves (selective accuracy and unflagged FN rate over ALL cases)
        s_full = np.where(dis, -1.0, np.minimum(mA, mB))
        scores = {"Full two-agent": s_full, "Margin A or B": np.minimum(mA, mB), "Margin A only": mA}
        n = len(y)
        for nm, sc in scores.items():
            accs, fns = [], []
            for c in GRID:
                kk = int(round(c * n))
                mm = sel_metrics(y, predA, topk_mask(sc, kk, rng))
                accs.append(mm["acc"]); fns.append(mm["fn"] / n)
            rc[label][nm].append((accs, fns))
        accs, fns = [], []
        for c in GRID:
            kk = int(round(c * n))
            mm = random_baseline(y, predA, kk, rng, R=20)
            accs.append(mm["acc"]); fns.append(mm["fn"] / n)
        rc[label]["Random"].append((accs, fns))

ms = pd.DataFrame(ms_rows)
ms.to_csv("results_multiseed.csv", index=False)
for label in ("unweighted", "balanced"):
    d = ms[ms.setting == label]
    print(f"\n[{label}]  n_seeds={len(d)}")
    summ = pd.DataFrame({
        "mean": d.drop(columns=["setting", "seed"]).mean(),
        "std": d.drop(columns=["setting", "seed"]).std(),
    })
    print(fmt(summ))
    dacc = d.full_acc - d.Amargin_acc
    dfn = d.full_fn - d.Amargin_fn
    print(f"  Full minus A-margin at same coverage: acc diff {dacc.mean():+.4f} (std {dacc.std():.4f}, "
          f"better in {(dacc > 0).sum()}/{len(d)} seeds); unflagged FN diff {dfn.mean():+.2f} "
          f"(fewer in {(dfn < 0).sum()}/{len(d)} seeds)")

# ---------------------------------------------------------------- 4. plot
plt.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3 (venue font rules)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
styles = {"Full two-agent": "-", "Margin A or B": "--", "Margin A only": "-.", "Random": ":"}
for nm, ls in styles.items():
    arr = np.array(rc["balanced"][nm])  # seeds x 2 x grid
    axes[0].plot(GRID, arr[:, 0, :].mean(0), ls, label=nm)
    axes[1].plot(GRID, arr[:, 1, :].mean(0), ls, label=nm)
axes[0].set_xlabel("Coverage (fraction decided automatically)")
axes[0].set_ylabel("Selective accuracy")
axes[1].set_xlabel("Coverage (fraction decided automatically)")
axes[1].set_ylabel("Unflagged false negatives / all cases")
axes[0].legend(fontsize=8)
for a in axes:
    a.grid(alpha=0.3)
fig.suptitle("Risk-coverage, class-balanced agents (mean of 20 seeds)", fontsize=10)
fig.tight_layout()
fig.savefig("risk_coverage.png", dpi=200)
fig.savefig("risk_coverage.pdf")
print("\nsaved risk_coverage.png / .pdf and results_*.csv")
