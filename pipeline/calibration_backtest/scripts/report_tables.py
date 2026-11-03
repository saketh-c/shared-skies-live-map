"""Print the REPORT.md tables from results.json (no hand-copied numbers)."""
import json, sys
r = json.load(open(sys.argv[1]))
lab = r["labels"]
def f(v, nd=2, sign=False):
    return "n/a" if v is None else (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")
for per in ("test", "train"):
    for p in ("P1", "P2"):
        t = r["tables"][per][p]
        rows = [(c, v) for c, v in t.items() if not c.startswith("_")]
        rows.sort(key=lambda cv: cv[1]["mae"])
        print(f"\n#### {per} {p} (n = {rows[0][1]['n']:,} Texas monitor-days)\n")
        print("| method | variant | bias | MAE | RMSE | r | mean daily spatial r | MAE, truth > 12 |")
        print("|---|---|---:|---:|---:|---:|---:|---:|")
        for c, v in rows:
            name = lab.get(c, c)
            print(f"| {name} | `{c}` | {f(v['bias'], sign=True)} | {f(v['mae'])} | {f(v['rmse'])} | {f(v['r'])} | {f(v['spatial_r_mean'])} | {f(v['mae_truth_gt12'])} |")
for p in ("P1", "P2"):
    t = r["tables"]["test"][p]
    print(f"\n#### region, test {p}\n\n| region | n | raw MAE (bias) | shipped MAE (bias) | monitors-only MAE |\n|---|---:|---:|---:|---:|")
    for reg, v in t["SHIPPED"]["region_mae"].items():
        m0 = t["M0"]["region_mae"][reg]; mo = t[r["tuned_variants"]["MO"]]["region_mae"][reg]
        print(f"| {reg} | {v['n']:,} | {m0['mae']:.2f} ({m0['bias']:+.2f}) | {v['mae']:.2f} ({v['bias']:+.2f}) | {mo['mae']:.2f} |")
    print(f"\n#### distance to nearest fitted monitor, test {p}\n\n| km | n | raw | shipped | monitors-only | daily Theil-Sen line |\n|---|---:|---:|---:|---:|---:|")
    for b, v in t["SHIPPED"]["dist_mae"].items():
        print(f"| {b} | {v['n']:,} | {t['M0']['dist_mae'][b]['mae']:.2f} | {v['mae']:.2f} | {t[r['tuned_variants']['MO']]['dist_mae'][b]['mae']:.2f} | {t['M2_ts']['dist_mae'][b]['mae']:.2f} |")
print("\n#### bootstrap\n\n| protocol | shipped minus | MAE diff | 95% CI |\n|---|---|---:|---|")
for b in r["bootstrap"]:
    print(f"| {b['protocol']} | {lab.get(b['b'], b['b'])} (`{b['b']}`) | {b['mae_diff']:+.3f} | [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] |")
print("\n", json.dumps(r["tables_all_monitors"]["test"], indent=0)[:600])
