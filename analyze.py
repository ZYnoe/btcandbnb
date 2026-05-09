"""Generate a Markdown analysis report + verdict plot from an outputs/ directory.

Usage:
    uv run python analyze.py outputs_quantum/
    uv run python analyze.py outputs_quantum/ --out report.md
    uv run python analyze.py outputs_quantum/ --no-plot

Reads:
    comparison_summary.csv, basic_stats.json, grid_search_results.csv,
    quantum_discrete_weight_results.csv, quantum_samples.csv  (all optional;
    missing files become "missing" in the report instead of crashing).

Writes:
    <outputs>/analysis_report.md
    <outputs>/verdict.png   (one combined figure: Sharpe-vs-BTC curve with
                             every method's choice marked + a ranked bar chart)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ---- helpers ------------------------------------------------------------

def _truthy(x) -> bool:
    """Pandas reads CSV booleans back as strings — normalize."""
    return str(x).strip().lower() in ("true", "1", "yes", "y")


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _load(outputs: Path) -> dict:
    out = {"path": outputs}

    p = outputs / "comparison_summary.csv"
    out["summary"] = pd.read_csv(p) if p.exists() else None

    p = outputs / "basic_stats.json"
    out["stats"] = json.loads(p.read_text()) if p.exists() else None

    p = outputs / "grid_search_results.csv"
    out["grid"] = pd.read_csv(p) if p.exists() else None

    p = outputs / "quantum_discrete_weight_results.csv"
    out["candidates"] = pd.read_csv(p) if p.exists() else None

    p = outputs / "quantum_samples.csv"
    if p.exists():
        df = pd.read_csv(p)
        out["samples"] = df if not df.empty else None
    else:
        out["samples"] = None

    return out


def _section(title: str) -> str:
    return f"\n## {title}\n\n"


# ---- sections -----------------------------------------------------------

def _completeness(d: dict) -> str:
    md = _section("1. Snapshot")
    files = [
        ("comparison_summary.csv", d["summary"], lambda x: f"{len(x)} rows"),
        ("basic_stats.json", d["stats"], lambda x: "OK"),
        ("grid_search_results.csv", d["grid"], lambda x: f"{len(x)} rows"),
        ("quantum_discrete_weight_results.csv", d["candidates"], lambda x: f"{len(x)} rows"),
        ("quantum_samples.csv", d["samples"], lambda x: f"{len(x)} samples"),
    ]
    rows = []
    for name, val, fn in files:
        rows.append(f"- `{name}`: {fn(val) if val is not None else '_missing or empty_'}")
    md += "\n".join(rows)

    if d["summary"] is not None:
        n = len(d["summary"])
        if n < 11:
            md += (
                f"\n\n> ⚠️ **Incomplete run**: comparison_summary has only {n}/11 expected rows. "
                f"The job was probably killed before all 6 quantum solvers finished. "
                f"Re-run with `QUANTUM_WEIGHT_STEP=0.1` for a 10-min full run.\n"
            )
    return md


def _data_section(d: dict) -> str:
    md = _section("2. Data Fundamentals")
    s = d["stats"]
    if s is None:
        return md + "_basic_stats.json missing._\n"
    md += f"- **Window**: {s['start']} → {s['end']} ({s['rows']} daily rows)\n"
    md += f"- **Tickers**: {', '.join(s['tickers'])}\n"
    md += f"- **Annualization factor**: {s['annualization_factor']}\n\n"
    md += "| Asset | μ_annual | σ_annual | Standalone Sharpe |\n|---|---|---|---|\n"
    for t in s["tickers"]:
        mu = s["annualized_return"][t]
        vol = s["annualized_volatility"][t]
        sharpe = mu / vol if vol > 0 else float("nan")
        md += f"| {t} | {mu:+.4f} ({mu*100:+.2f}%) | {vol:.4f} ({vol*100:.2f}%) | {sharpe:.4f} |\n"

    rho = s["correlation"][0][1]
    md += f"\n- **Correlation**: ρ = {rho:+.4f}\n"

    # Domination prediction — useful prior to interpreting the optimizer
    tickers = s["tickers"]
    if len(tickers) == 2:
        a, b = tickers
        mu_a, mu_b = s["annualized_return"][a], s["annualized_return"][b]
        v_a, v_b = s["annualized_volatility"][a], s["annualized_volatility"][b]
        if mu_a > mu_b and v_a < v_b:
            md += f"\n> **{a} strictly dominates {b}** (higher μ, lower σ). Expect optimum at corner ≈ 100% {a}.\n"
        elif mu_b > mu_a and v_b < v_a:
            md += f"\n> **{b} strictly dominates {a}**. Expect optimum at corner ≈ 100% {b}.\n"
        else:
            md += f"\n> **Neither asset dominates** — optimum likely interior, depends on correlation and risk aversion.\n"
    return md


def _methods_table(d: dict) -> str:
    md = _section("3. All Methods, ranked by Sharpe")
    df = d["summary"]
    if df is None or df.empty:
        return md + "_comparison_summary.csv missing or empty._\n"

    df = df.copy()
    df["_ok"] = df["success"].apply(_truthy)
    success = df[df["_ok"]].copy()
    failed = df[~df["_ok"]].copy()

    if not success.empty:
        success["_sharpe"] = success["sharpe_ratio"].apply(_safe_float)
        success = success.sort_values("_sharpe", ascending=False)
        md += "| Rank | Method | Solver | BTC | Sharpe | Vol | MaxDD | FinalCum | Runtime |\n"
        md += "|---|---|---|---|---|---|---|---|---|\n"
        for i, (_, r) in enumerate(success.iterrows(), 1):
            md += (
                f"| {i} | {r['method']} | {r['solver']} | "
                f"{_safe_float(r['btc_weight']):.3f} | "
                f"{_safe_float(r['sharpe_ratio']):.4f} | "
                f"{_safe_float(r['annual_volatility']):.4f} | "
                f"{_safe_float(r['max_drawdown']):+.4f} | "
                f"{_safe_float(r['final_cumulative_return']):+.4f} | "
                f"{_safe_float(r['runtime_seconds']) * 1000:.1f} ms |\n"
            )

    if not failed.empty:
        md += "\n### Failed runs\n\n"
        for _, r in failed.iterrows():
            msg = r.get("error_message", "") or "(no error_message)"
            md += f"- `{r['method']} / {r['solver']}` — {msg}\n"
    return md


def _cross_validation(d: dict) -> str:
    md = _section("4. Cross-validation: quantum-exact ↔ classical-grid")
    df = d["summary"]
    if df is None:
        return md + "_summary missing._\n"

    md += "Sanity check: the quantum **exact** solver and the classical **grid** must agree (within step granularity).\n\n"

    grid = df[df["method"].fillna("").str.contains("Classic Grid Search")]
    qbin = df[(df["method"].fillna("").str.contains("Quantum Binary")) & (df["solver"] == "exact")]
    qdis = df[(df["method"].fillna("").str.contains("Quantum Discrete")) & (df["solver"] == "exact")]

    rows = []
    if not grid.empty:
        g = grid.iloc[0]
        rows.append(("Classical Grid", _safe_float(g["btc_weight"]), _safe_float(g["sharpe_ratio"])))
    if not qbin.empty:
        q = qbin.iloc[0]
        rows.append(("Quantum Binary exact", _safe_float(q["btc_weight"]), _safe_float(q["sharpe_ratio"])))
    if not qdis.empty:
        q = qdis.iloc[0]
        rows.append(("Quantum Discrete exact", _safe_float(q["btc_weight"]), _safe_float(q["sharpe_ratio"])))

    md += "| Source | BTC | Sharpe |\n|---|---|---|\n"
    for label, btc, sh in rows:
        md += f"| {label} | {btc:.4f} | {sh:.6f} |\n"

    if not grid.empty and not qdis.empty:
        g_btc = _safe_float(grid.iloc[0]["btc_weight"])
        q_btc = _safe_float(qdis.iloc[0]["btc_weight"])
        nearest_005 = round(g_btc * 20) / 20
        if abs(q_btc - nearest_005) < 1e-6:
            md += (
                f"\n✓ **PASS** — Quantum Discrete picked BTC = {q_btc:.2f}, the nearest "
                f"0.05-step neighbour of the classical grid optimum BTC = {g_btc:.3f}. "
                "QUBO encoding (μ, Σ, one-hot constraint) is verified.\n"
            )
        else:
            md += (
                f"\n⚠️ **Mismatch** — Quantum Discrete BTC = {q_btc:.3f}, "
                f"but the nearest 0.05 step to grid {g_btc:.3f} is {nearest_005:.2f}. "
                "Investigate the Σ off-diagonal coefficient or the linear/quadratic mapping.\n"
            )
    return md


def _approx_quality(d: dict) -> str:
    md = _section("5. Approximate solvers — QAOA / SamplingVQE")
    df = d["summary"]
    if df is None:
        return md + "_summary missing._\n"

    df = df.copy()
    df["_ok"] = df["success"].apply(_truthy)

    approx = df[df["solver"].isin(["qaoa", "sampling_vqe"]) & df["_ok"]].copy()
    exact_rows = df[(df["solver"] == "exact") & df["_ok"]].copy()
    if approx.empty:
        return md + "_No QAOA / SamplingVQE successful rows._\n"

    md += "How close did approximate solvers come to the exact answer of the same problem family?\n\n"
    md += "| Method | Solver | BTC | Sharpe | Δobjective vs exact | Note |\n"
    md += "|---|---|---|---|---|---|\n"
    for _, r in approx.iterrows():
        kind = "Binary" if "Binary" in str(r["method"]) else "Discrete"
        ex = exact_rows[exact_rows["method"].fillna("").str.contains(kind)]
        if not ex.empty:
            d_obj = _safe_float(r["objective"]) - _safe_float(ex.iloc[0]["objective"])
            d_str = f"{d_obj:+.6f}"
        else:
            d_str = "N/A"
        note = (r.get("note", "") or "").replace("\n", " ")
        md += (
            f"| {r['method']} | {r['solver']} | "
            f"{_safe_float(r['btc_weight']):.3f} | "
            f"{_safe_float(r['sharpe_ratio']):.4f} | {d_str} | {note} |\n"
        )

    md += (
        "\n*A non-zero Δobjective is normal — QAOA/SamplingVQE are heuristic. "
        "If they converged to the same bitstring as exact, the corresponding row will have Δ = 0.*\n"
    )
    return md


def _runtime_section(d: dict) -> str:
    md = _section("6. Runtime: classical vs quantum")
    df = d["summary"]
    if df is None:
        return md + "_summary missing._\n"

    df = df.copy()
    df["_ok"] = df["success"].apply(_truthy)
    success = df[df["_ok"]].copy()
    if success.empty:
        return md + "_no successful rows._\n"

    classical = success[success["method"].fillna("").str.startswith("Classic")]
    if classical.empty:
        return md + "_no classical rows; cannot establish baseline._\n"

    success["_rt"] = success["runtime_seconds"].apply(_safe_float)
    baseline = max(classical["runtime_seconds"].apply(_safe_float).min(), 1e-6)

    md += f"**Baseline** (fastest classical solver): {baseline*1000:.2f} ms.\n\n"
    md += "| Method | Solver | Runtime | × baseline |\n|---|---|---|---|\n"
    for _, r in success.iterrows():
        rt = _safe_float(r["runtime_seconds"])
        if rt > 0:
            ratio = rt / baseline
            ratio_str = f"{ratio:.0f}×" if ratio >= 10 else f"{ratio:.2f}×"
        else:
            ratio_str = "—"
        md += f"| {r['method']} | {r['solver']} | {rt*1000:.2f} ms | {ratio_str} |\n"

    md += (
        "\nFor a 2-asset, 1-D continuous problem, classical Mean-Variance is essentially "
        "free; quantum solvers pay 3–5 orders of magnitude more wall time to reach the same "
        "answer. This is the cost of the QUBO mapping, not a Qiskit defect.\n"
    )
    return md


def _landscape_section(d: dict) -> str:
    md = _section("7. Discrete-weight landscape")
    cands = d["candidates"]
    if cands is None or cands.empty:
        return md + "_quantum_discrete_weight_results.csv missing._\n"

    cands = cands.copy()
    for col in ("btc_weight", "annual_return", "annual_volatility", "sharpe_ratio",
                "max_drawdown", "annual_variance", "objective"):
        cands[col] = cands[col].apply(_safe_float)

    sharpe_max_idx = cands["sharpe_ratio"].idxmax()
    vol_min_idx = cands["annual_volatility"].idxmin()
    mdd_max_idx = cands["max_drawdown"].idxmax()  # least negative

    md += f"- {len(cands)} discrete candidates, BTC ∈ {{0.00, …, 1.00}}.\n"
    md += (
        f"- **Highest Sharpe**: BTC = {cands.loc[sharpe_max_idx, 'btc_weight']:.2f} "
        f"→ {cands.loc[sharpe_max_idx, 'sharpe_ratio']:.4f}\n"
    )
    md += (
        f"- **Lowest volatility**: BTC = {cands.loc[vol_min_idx, 'btc_weight']:.2f} "
        f"→ σ = {cands.loc[vol_min_idx, 'annual_volatility']:.4f}\n"
    )
    md += (
        f"- **Smallest drawdown**: BTC = {cands.loc[mdd_max_idx, 'btc_weight']:.2f} "
        f"→ MaxDD = {cands.loc[mdd_max_idx, 'max_drawdown']:+.4f}\n"
    )

    s_max = cands.loc[sharpe_max_idx, "sharpe_ratio"]
    near = cands[cands["sharpe_ratio"] >= s_max - 0.01]
    if len(near) > 1:
        md += (
            f"- **Sharpe plateau** (within 0.01 of max): BTC ∈ "
            f"[{near['btc_weight'].min():.2f}, {near['btc_weight'].max():.2f}] — "
            f"{len(near)} candidates virtually equivalent.\n"
        )

    if sharpe_max_idx != vol_min_idx:
        md += (
            f"\n> **Diversification paradox**: lowest-vol BTC = "
            f"{cands.loc[vol_min_idx, 'btc_weight']:.2f} has Sharpe "
            f"{cands.loc[vol_min_idx, 'sharpe_ratio']:.4f}, "
            f"BELOW the max Sharpe {s_max:.4f}. Mean dominates variance reduction in this window.\n"
        )
    return md


def _samples_section(d: dict) -> str:
    md = _section("8. QAOA / SamplingVQE measurement distribution")
    s = d["samples"]
    if s is None or s.empty:
        return md + (
            "_quantum_samples.csv is empty — no top-k samples were captured. "
            "This usually means MinimumEigenOptimizer's result didn't expose `.samples` "
            "in the installed Qiskit version, or the quantum solvers were skipped._\n"
        )
    md += "Top sampled bitstrings (by probability):\n\n"
    md += "| Solver | Bitstring | BTC weight | Probability | Objective |\n|---|---|---|---|---|\n"
    for _, r in s.head(10).iterrows():
        md += (
            f"| {r.get('solver','?')} | {r.get('bitstring','?')} | "
            f"{r.get('btc_weight','?')} | {r.get('probability','?')} | "
            f"{r.get('objective','?')} |\n"
        )
    if "probability" in s.columns:
        max_prob = s["probability"].apply(_safe_float).max()
        if max_prob > 0.9:
            md += f"\n**Sharp peak** (max p = {max_prob:.3f}): QAOA converged on a single bitstring.\n"
        elif max_prob > 0.5:
            md += f"\n**Moderate peak** (max p = {max_prob:.3f}): preference exists but tail is long.\n"
        else:
            md += (
                f"\n**Diffuse distribution** (max p = {max_prob:.3f}): QAOA did not converge — "
                "try `--qaoa-reps 3 --qaoa-shots 4096`.\n"
            )
    return md


def _caveats() -> str:
    return _section("9. Caveats — read before drawing conclusions") + (
        "- **In-sample optimum**: every \"best weight\" reported here is in-sample on the "
        "historical window. There is no guarantee it remains optimal forward.\n"
        "- **Crypto bull windows distort Sharpe**: a typical equity Sharpe is 0.4–0.8. "
        "Sharpe ≥ 1.5 in crypto reflects an unusually favourable window, not a sustainable "
        "risk-adjusted return.\n"
        "- **Window-sensitivity**: shifting start/end dates by 6 months can flip the optimum. "
        "Run a rolling-window sensitivity analysis before trusting any recommendation.\n"
        "- **Quantum results don't predict prices**: QAOA / SamplingVQE here only solve the "
        "QUBO under the *given* μ and Σ. They tell you which weights minimize risk-adjusted "
        "cost on past data, not what the market will do tomorrow.\n"
        "- **NOT investment advice** — this tool is for research and education only.\n"
    )


# ---- verdict plot -------------------------------------------------------

def _verdict_plot(d: dict, output_path: Path) -> None:
    grid = d["grid"]
    summary = d["summary"]
    if grid is None or summary is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # LEFT — Sharpe vs BTC weight, with all methods marked
    ax = axes[0]
    ax.plot(grid["btc_weight"], grid["sharpe_ratio"], color="#aaaaaa", lw=2, zorder=1, label="Grid (continuous)")

    df = summary.copy()
    df["_ok"] = df["success"].apply(_truthy)
    success = df[df["_ok"]]

    palette = {"Classic": "tab:blue", "Quantum": "tab:orange", "Benchmark": "tab:gray"}
    markers = {"Classic": "o", "Quantum": "*", "Benchmark": "s"}
    sizes = {"Classic": 90, "Quantum": 200, "Benchmark": 90}
    seen = set()
    for _, r in success.iterrows():
        m = str(r["method"])
        kind = "Classic" if m.startswith("Classic") else ("Quantum" if m.startswith("Quantum") else "Benchmark")
        ax.scatter(
            _safe_float(r["btc_weight"]), _safe_float(r["sharpe_ratio"]),
            color=palette[kind], marker=markers[kind], s=sizes[kind],
            edgecolor="black", linewidth=0.6, zorder=3,
            label=kind if kind not in seen else None,
        )
        seen.add(kind)

    ax.set_title("Sharpe vs BTC weight — every method on one axis")
    ax.set_xlabel("BTC weight")
    ax.set_ylabel("Sharpe ratio")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    # RIGHT — horizontal bar chart of all methods
    ax = axes[1]
    if not success.empty:
        s2 = success.copy()
        s2["_sharpe"] = s2["sharpe_ratio"].apply(_safe_float)
        s2 = s2.sort_values("_sharpe", ascending=True)
        labels = s2["method"].astype(str) + " · " + s2["solver"].astype(str)
        colors = []
        for m in s2["method"]:
            mm = str(m)
            colors.append(
                "tab:blue" if mm.startswith("Classic")
                else ("tab:orange" if mm.startswith("Quantum") else "tab:gray")
            )
        ax.barh(labels, s2["_sharpe"], color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Sharpe ratio")
        ax.set_title("All methods, ranked")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Verdict — {output_path.parent.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# ---- main ---------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("outputs_dir", type=Path,
                   help="Path to outputs_quantum/ or outputs_classical/")
    p.add_argument("--out", type=Path, default=None,
                   help="Output Markdown path (default: <outputs>/analysis_report.md)")
    p.add_argument("--no-plot", action="store_true", help="Skip the verdict plot")
    args = p.parse_args(argv)

    if not args.outputs_dir.is_dir():
        print(f"error: {args.outputs_dir} is not a directory", file=sys.stderr)
        return 2

    out_md = args.out or (args.outputs_dir / "analysis_report.md")
    out_png = args.outputs_dir / "verdict.png"

    d = _load(args.outputs_dir)
    sections = [
        f"# Portfolio Optimization Analysis Report\n\n_Generated from `{args.outputs_dir}/`._\n",
        _completeness(d),
        _data_section(d),
        _methods_table(d),
        _cross_validation(d),
        _approx_quality(d),
        _runtime_section(d),
        _landscape_section(d),
        _samples_section(d),
        _caveats(),
    ]
    out_md.write_text("\n".join(sections))
    print(f"wrote {out_md}")

    if not args.no_plot:
        try:
            _verdict_plot(d, out_png)
            print(f"wrote {out_png}")
        except Exception as e:  # noqa: BLE001
            print(f"plot failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
