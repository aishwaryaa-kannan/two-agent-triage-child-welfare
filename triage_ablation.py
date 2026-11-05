"""
Multi-seed ablation of the flagging signals (class-balanced agents, margin 0.1, 20 seeds).
Each rule is compared with an A-confidence-only baseline and a random baseline that keep
the same number of cases (same coverage).
"""
from triage_lib import *

rows = []
for seed in range(20):
    y, pA, pB = run_agents(seed, seed, "balanced")
    predA, dis, mA, mB, rules = flag_rules(pA, pB)
    rng = np.random.default_rng(seed)
    n = len(y)
    for name, flag in rules.items():
        keep = ~flag
        k = int(keep.sum())
        m = sel_metrics(y, predA, keep)
        base = sel_metrics(y, predA, topk_mask(mA, k, rng))
        rnd = random_baseline(y, predA, k, rng, R=50)
        rows.append(dict(rule=name, seed=seed, cov=m["cov"], acc=m["acc"], f1=m["f1"], rec=m["rec"],
                         fn_per_case=m["fn"] / n, Amargin_acc=base["acc"], Amargin_fn_per_case=base["fn"] / n,
                         random_acc=rnd["acc"], random_fn_per_case=rnd["fn"] / n))
d = pd.DataFrame(rows)
d.to_csv("results_ablation_multiseed.csv", index=False)

order = ["Margin A only (Chow)", "Disagreement only", "Margin A or B", "Full (disagree or margin)"]
cols = ["cov", "acc", "f1", "rec", "fn_per_case", "Amargin_acc", "Amargin_fn_per_case", "random_acc"]
summ = d.groupby("rule")[cols].agg(["mean", "std"]).loc[order].round(3)
print(summ.to_string())
for r in order:
    s = d[d.rule == r]
    da, df_ = s.acc - s.Amargin_acc, s.fn_per_case - s.Amargin_fn_per_case
    print(f"{r:28s} acc diff vs A-confidence at same coverage {da.mean():+.4f} "
          f"(better {int((da > 1e-12).sum())}/20); FN-rate diff {df_.mean():+.4f} (lower {int((df_ < -1e-12).sum())}/20)")
