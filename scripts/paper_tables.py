"""Generate the paper's tables and number macros from the result files.

    python scripts/paper_tables.py     # paper/tables/*.tex, paper/tables/numbers.tex,
                                       # paper/data/macro_sources.json, paper/data/*.json snapshots

Every number in the paper is a macro \\V{key} defined in paper/tables/numbers.tex. For each key,
paper/data/macro_sources.json records the printed text, the raw value, the source file, the field
(a query that scripts/paper_claims.py re-reads by its own code path, or "derived: ..." for values
computed from several inputs) and the scale between raw and printed values (100 for percentages).
Nothing is typed by hand. Git-ignored inputs (the preprocessing report, live-clip landmarks) are
snapshotted into paper/data/ when present, so the tables rebuild on a fresh clone.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
DATA = PAPER / "data"
ALPHA = 0.05

EXPERIMENTS = [  # (experiment, short name, model, features, recipe, candidate in model_selection)
    ("gru_signer_independent", "GRU", "gru", "v1", "v1", "gru_v1"),
    ("transformer_signer_independent", "Transformer", "transformer", "v1", "v1", "transformer_v1"),
    ("cnn_signer_independent", "CNN", "cnn", "v1", "v1", "cnn_v1"),
    ("b_cnn_v2feat_v1recipe", "CNN", "cnn", "v2", "v1", "cnn_v2feat_v1recipe"),
    ("c_cnn_v2feat_v2recipe", "CNN", "cnn", "v2", "v2", "cnn_v2feat_v2recipe"),
    ("d_hybrid_v2feat_v2recipe", "Hybrid", "hybrid", "v2", "v2", "hybrid_v2"),
]
SHORT = {e[0]: e for e in EXPERIMENTS}
CUMULATIVE = [
    "c0_as_shipped",
    "c1_model",
    "c2_hands",
    "c3_aspect",
    "c4_time",
    "c5_segmenter",
    "c6_final",
]
LEAVE_ONE_OUT = [
    "x_hands",
    "x_mirrored",
    "x_mirrored_label",
    "x_aspect_off",
    "x_aspect_calibrated",
    "x_index",
    "x_presence",
    "x_v1_model",
]
STEP_NAMES = {
    "c0_as_shipped": "As shipped (v1 CNN)",
    "c1_model": "+ v2 hybrid model",
    "c2_hands": "+ hand assignment (a)",
    "c3_aspect": "+ aspect from frame (c)",
    "c4_time": "+ time resampling (d)",
    "c5_segmenter": "+ motion segmenter (e)",
    "c6_final": "+ calibration (f): final",
    "x_hands": "final, old hand mapping",
    "x_mirrored": "final, mirrored frame (b)",
    "x_mirrored_label": "final, mirrored + old mapping",
    "x_aspect_off": "final, no aspect corr.",
    "x_aspect_calibrated": "final, body-ratio aspect",
    "x_index": "final, index resampling",
    "x_presence": "final, presence segmenter",
    "x_v1_model": "final, v1 CNN",
    "c0_head_build": "Shipped build",
}


# --------------------------------------------------------------------------------------------- io


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def row(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    found = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    if len(found) != 1:
        raise KeyError(f"expected one row for {match}, found {len(found)}")
    return found[0]


# ------------------------------------------------------------------------------------- macros


class Macros:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def put(
        self, key: str, raw: Any, text: str, source: Path | str, field: str, scale: float = 1.0
    ) -> str:
        if key in self.items:
            raise ValueError(f"macro {key} defined twice")
        src = rel(source) if isinstance(source, Path) else source
        self.items[key] = {"text": text, "raw": raw, "source": src, "field": field, "scale": scale}
        return text

    def num(
        self,
        key: str,
        raw: float,
        decimals: int,
        source: Path | str,
        field: str,
        scale: float = 1.0,
        signed: bool = False,
    ) -> str:
        value = float(raw) * scale
        text = f"{value:+.{decimals}f}" if signed else f"{value:,.{decimals}f}"
        return self.put(key, float(raw), text.replace("-", "$-$"), source, field, scale)

    def pct(
        self,
        key: str,
        raw: float,
        decimals: int,
        source: Path | str,
        field: str,
        signed: bool = False,
    ) -> str:
        return self.num(key, raw, decimals, source, field, scale=100.0, signed=signed)

    def ci(
        self,
        key: str,
        low: float,
        high: float,
        decimals: int,
        source: Path | str,
        field: str,
        scale: float = 100.0,
    ) -> str:
        a, b = f"{low * scale:.{decimals}f}", f"{high * scale:.{decimals}f}"
        text = f"[{a}, {b}]".replace("-", "$-$")
        return self.put(key, [float(low), float(high)], text, source, field, scale)

    def p(self, key: str, raw: float, source: Path | str, field: str) -> str:
        text = "$<$0.001" if raw < 0.001 else f"{raw:.3f}"
        return self.put(key, float(raw), text, source, field)

    def sig(self, key: str, p_holm: float, source: Path | str, field: str) -> str:
        return self.put(
            key,
            float(p_holm),
            "significant" if p_holm < ALPHA else "not significant",
            source,
            field,
        )

    def word(self, key: str, value: str, source: Path | str, field: str) -> str:
        return self.put(key, value, value, source, field)

    def write(self) -> None:
        lines = [
            "% Generated by scripts/paper_tables.py -- do not edit by hand.",
            "% Each number quoted in the paper is \\V{key}; sources: paper/data/macro_sources.json.",
        ]
        for key in sorted(self.items):
            lines.append(
                f"\\expandafter\\newcommand\\csname v@{key}\\endcsname{{{self.items[key]['text']}}}"
            )
        (TABLES / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (DATA / "macro_sources.json").write_text(
            json.dumps(self.items, indent=1) + "\n", encoding="utf-8"
        )


def tex_escape(text: str) -> str:
    return text.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def write_table(name: str, body: str) -> None:
    # The \typeout marker names the table in the LaTeX log, next to any box warning it causes.
    (TABLES / f"{name}.tex").write_text(
        "% Generated by scripts/paper_tables.py -- do not edit.\n"
        f"\\typeout{{[table {name}]}}\n" + body + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------------------- data


def dataset(m: Macros) -> None:
    report_path = ROOT / "data" / "processed_v2" / "preprocess_report.json"
    snapshot = DATA / "dataset_stats.json"
    if report_path.exists():
        report = read_json(report_path)
        snap = {
            "source": rel(report_path),
            "counts": report["counts"],
            "per_sign_kept": report["per_sign_kept"],
            "mirrored_fraction": report["mirrored_fraction"],
            "landmarks": report["landmarks"],
            "num_frames": report["config"]["preprocess"]["num_frames"],
            "min_hand_frames": report["config"]["preprocess"]["min_hand_frames"],
            "v1_pose_indices": read_json(ROOT / "data" / "processed" / "preprocess_report.json")[
                "config"
            ]["preprocess"]["pose_indices"],
        }
        snapshot.write_text(json.dumps(snap, indent=1) + "\n", encoding="utf-8")
    snap = read_json(snapshot)
    m.num("data.total", snap["counts"]["total"], 0, snapshot, "counts.total")
    m.num("data.kept", snap["counts"]["kept"], 0, snapshot, "counts.kept")
    m.num("data.excluded", snap["counts"]["excluded"], 0, snapshot, "counts.excluded")
    m.num("data.signmin", snap["per_sign_kept"]["min"], 0, snapshot, "per_sign_kept.min")
    m.num("data.signmax", snap["per_sign_kept"]["max"], 0, snapshot, "per_sign_kept.max")
    m.pct("data.mirroredpct", snap["mirrored_fraction"], 1, snapshot, "mirrored_fraction")
    m.num("data.frames", snap["num_frames"], 0, snapshot, "num_frames")
    m.num("data.minhand", snap["min_hand_frames"], 0, snapshot, "min_hand_frames")
    m.num(
        "data.v2points", snap["landmarks"]["num_landmarks"], 0, snapshot, "landmarks.num_landmarks"
    )
    m.num(
        "data.v2pose",
        len(snap["landmarks"]["pose_indices"]),
        0,
        snapshot,
        "derived: len(landmarks.pose_indices)",
    )
    m.num("data.v1pose", len(snap["v1_pose_indices"]), 0, snapshot, "derived: len(v1_pose_indices)")
    hand = read_json(ROOT / "features_spec.json")["selection"]["hand_points"]
    m.num(
        "data.v1points",
        2 * hand + len(snap["v1_pose_indices"]),
        0,
        snapshot,
        "derived: 2 x selection.hand_points (features_spec.json) + len(v1_pose_indices)",
    )
    spec_path = ROOT / "features_spec.json"
    spec = read_json(spec_path)
    counts = spec["landmark_types"]["counts"]
    m.num("data.lmface", counts["face"], 0, spec_path, "landmark_types.counts.face")
    m.num("data.lmpose", counts["pose"], 0, spec_path, "landmark_types.counts.pose")
    m.num("data.lmhand", counts["left_hand"], 0, spec_path, "landmark_types.counts.left_hand")
    m.num("data.lmtotal", sum(counts.values()), 0, spec_path, "derived: sum(landmark_types.counts)")
    split_path = (
        ROOT
        / "results"
        / "runs"
        / "d_hybrid_v2feat_v2recipe"
        / "seed_0"
        / "splits_signer_independent.json"
    )
    split = read_json(split_path)
    for part in ("train", "val", "test"):
        m.num(
            f"data.{part}seq",
            split["summary"][part]["sequences"],
            0,
            split_path,
            f"summary.{part}.sequences",
        )
        m.num(
            f"data.{part}signers",
            len(split["participants"][part]),
            0,
            split_path,
            f"derived: len(participants.{part})",
        )
    m.num(
        "data.signers",
        sum(len(v) for v in split["participants"].values()),
        0,
        split_path,
        "derived: sum of len(participants.*)",
    )
    m.num("data.signs", split["summary"]["train"]["labels"], 0, split_path, "summary.train.labels")
    # The v2 feature block, computed from the code that builds it (inside the ONNX graph).
    from edgevision.layout import InputLayout
    from edgevision.models import features_v2

    fv2 = ROOT / "src" / "edgevision" / "models" / "features_v2.py"
    layout = InputLayout.for_pose_indices(snap["landmarks"]["pose_indices"])
    m.num(
        "data.v2feat",
        features_v2.v2_feature_dim(layout),
        0,
        fv2,
        "derived: v2_feature_dim(v2 layout)",
    )
    m.num("data.pairdist", len(features_v2.HAND_PAIRS), 0, fv2, "derived: len(HAND_PAIRS)")
    m.num(
        "data.facedist",
        len(features_v2.BODY_POINTS) * len(features_v2.BODY_TARGETS),
        0,
        fv2,
        "derived: len(BODY_POINTS) x len(BODY_TARGETS)",
    )


# ------------------------------------------------------------------------------------- models


def models(m: Macros) -> None:
    sel = ROOT / "results" / "tables" / "selection"
    test_path, val_path, ens_path = (
        sel / "test_fp32_summary.csv",
        sel / "val_summary.csv",
        sel / "test_ensemble.csv",
    )
    test, val, ens = read_csv(test_path), read_csv(val_path), read_csv(ens_path)
    msel_path = ROOT / "results" / "model_selection.csv"
    msel = read_csv(msel_path)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lllrrrrrr}",
        "\\toprule",
        "Model & Features & Recipe & Params (M) & Val top-1 & Test top-1 & Test top-3 & ONNX (MB) & Browser p95 (ms) \\\\",
        "\\midrule",
    ]
    for exp, short, _model, feat, recipe, cand in EXPERIMENTS:
        t, v, s = row(test, experiment=exp), row(val, experiment=exp), row(msel, candidate=cand)
        k = {
            "gru_signer_independent": "gru",
            "transformer_signer_independent": "tf",
            "cnn_signer_independent": "cnn",
            "b_cnn_v2feat_v1recipe": "b",
            "c_cnn_v2feat_v2recipe": "c",
            "d_hybrid_v2feat_v2recipe": "d",
        }[exp]
        top1 = m.pct(
            f"m.{k}.top1",
            float(t["test_top1_mean"]),
            1,
            test_path,
            f"test_top1_mean|experiment={exp}",
        )
        top1sd = m.pct(
            f"m.{k}.top1sd",
            float(t["test_top1_std"]),
            1,
            test_path,
            f"test_top1_std|experiment={exp}",
        )
        top3 = m.pct(
            f"m.{k}.top3",
            float(t["test_top3_mean"]),
            1,
            test_path,
            f"test_top3_mean|experiment={exp}",
        )
        top3sd = m.pct(
            f"m.{k}.top3sd",
            float(t["test_top3_std"]),
            1,
            test_path,
            f"test_top3_std|experiment={exp}",
        )
        vtop1 = m.pct(
            f"m.{k}.val", float(v["val_top1_mean"]), 1, val_path, f"val_top1_mean|experiment={exp}"
        )
        params = m.num(
            f"m.{k}.params",
            float(v["params"]) / 1e6,
            2,
            val_path,
            f"params|experiment={exp}",
            scale=1.0,
        )
        m.items[f"m.{k}.params"]["raw"] = float(v["params"])
        m.items[f"m.{k}.params"]["scale"] = 1e-6
        size = m.num(f"m.{k}.mb", float(s["onnx_mb"]), 1, msel_path, f"onnx_mb|candidate={cand}")
        p95 = m.num(
            f"m.{k}.p95",
            float(s["latency_p95_ms"]),
            1,
            msel_path,
            f"latency_p95_ms|candidate={cand}",
        )
        m.num(
            f"m.{k}.p50",
            float(s["latency_p50_ms"]),
            1,
            msel_path,
            f"latency_p50_ms|candidate={cand}",
        )
        m.num(
            f"m.{k}.trainmin",
            float(v["train_seconds_mean"]) / 60,
            0,
            val_path,
            f"train_seconds_mean|experiment={exp}",
            scale=1.0,
        )
        m.items[f"m.{k}.trainmin"]["raw"] = float(v["train_seconds_mean"])
        m.items[f"m.{k}.trainmin"]["scale"] = 1 / 60
        lines.append(
            f"{short} & {feat} & {recipe} & {params} & {vtop1} & "
            f"{top1}$\\pm${top1sd} & {top3}$\\pm${top3sd} & {size} & {p95} \\\\"
        )
    e = row(ens, experiment="d_hybrid_v2feat_v2recipe")
    m.pct(
        "m.d.ens.top1",
        float(e["test_top1"]),
        1,
        ens_path,
        "test_top1|experiment=d_hybrid_v2feat_v2recipe",
    )
    m.pct(
        "m.d.ens.top3",
        float(e["test_top3"]),
        1,
        ens_path,
        "test_top3|experiment=d_hybrid_v2feat_v2recipe",
    )
    ens_sel = row(msel, candidate="hybrid_v2_ensemble_fp16")
    m.num(
        "m.d.ens.mb",
        float(ens_sel["onnx_mb"]),
        1,
        msel_path,
        "onnx_mb|candidate=hybrid_v2_ensemble_fp16",
    )
    m.num(
        "m.d.ens.p95",
        float(ens_sel["latency_p95_ms"]),
        1,
        msel_path,
        "latency_p95_ms|candidate=hybrid_v2_ensemble_fp16",
    )
    fp16 = row(msel, candidate="hybrid_v2_fp16")
    m.num("m.ship.mb", float(fp16["onnx_mb"]), 1, msel_path, "onnx_mb|candidate=hybrid_v2_fp16")
    m.num(
        "m.ship.p50",
        float(fp16["latency_p50_ms"]),
        1,
        msel_path,
        "latency_p50_ms|candidate=hybrid_v2_fp16",
    )
    m.num(
        "m.ship.p95",
        float(fp16["latency_p95_ms"]),
        1,
        msel_path,
        "latency_p95_ms|candidate=hybrid_v2_fp16",
    )
    sd = row(test, experiment="transformer_signer_dependent")
    m.pct(
        "m.dep.tf.top1",
        float(sd["test_top1_mean"]),
        1,
        test_path,
        "test_top1_mean|experiment=transformer_signer_dependent",
    )
    sd_cnn = row(test, experiment="cnn_signer_dependent")
    m.pct(
        "m.dep.cnn.top1",
        float(sd_cnn["test_top1_mean"]),
        1,
        test_path,
        "test_top1_mean|experiment=cnn_signer_dependent",
    )
    summary_path = (
        ROOT / "results" / "runs" / "d_hybrid_v2feat_v2recipe" / "seed_0" / "summary.json"
    )
    summary = read_json(summary_path)
    m.word("m.gpu", summary["device_name"], summary_path, "device_name")
    m.num("m.d.bestepoch", summary["best_epoch"], 0, summary_path, "best_epoch")
    pd_, pc_ = (
        float(row(val, experiment=e)["params"])
        for e in ("d_hybrid_v2feat_v2recipe", "c_cnn_v2feat_v2recipe")
    )
    m.num(
        "m.paramratio",
        pd_ / pc_,
        1,
        val_path,
        "derived: params of the hybrid / params of the v2 CNN",
    )
    td_, tc_ = (
        float(row(val, experiment=e)["train_seconds_mean"])
        for e in ("d_hybrid_v2feat_v2recipe", "c_cnn_v2feat_v2recipe")
    )
    m.num(
        "m.timeratio",
        td_ / tc_,
        1,
        val_path,
        "derived: mean training time of the hybrid / of the v2 CNN",
    )
    m.num("m.d.epochsrun", summary["epochs_run"], 0, summary_path, "epochs_run")
    m.put(
        "repro.train",
        summary["run_info"]["git_commit"],
        summary["run_info"]["git_commit"][:7],
        summary_path,
        "run_info.git_commit",
    )
    env_path = ROOT / "results" / "bench" / "bench_env.json"
    commit = read_json(env_path)["run_info"]["git_commit"]
    m.put("repro.base", commit, commit[:7], env_path, "run_info.git_commit")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Signer-independent results: validation and test top-1/top-3 (\\%, mean $\\pm$ sd over "
        "seeds 0--2), parameter count, ONNX size and one classifier call in Chrome on the i3-1115G4 "
        "(onnxruntime-web, wasm SIMD, 1 thread, p95; seed-0 file, fp32). The shipped file is the "
        "hybrid in fp16 (Section~\\ref{sec:efficiency}).}",
        "\\label{tab:models}",
        "\\end{table*}",
    ]
    write_table("models", "\n".join(lines))


def comparisons(m: Macros) -> None:
    path = ROOT / "results" / "test_analysis" / "model_comparisons.csv"
    rows = read_csv(path)
    keys = {
        "v1: GRU vs CNN": "gru",
        "v1: transformer vs CNN": "tf",
        "v2 features (CNN, v1 recipe)": "feat",
        "v2 recipe (CNN, v2 features)": "recipe",
        "hybrid architecture (v2 features, v2 recipe)": "arch",
        "shipped hybrid vs v1 CNN": "ship",
    }
    short = {  # row labels: the change in the first model against the second
        "gru": "GRU vs. CNN",
        "tf": "Transf. vs. CNN",
        "feat": "+ v2 features",
        "recipe": "+ v2 recipe",
        "arch": "+ hybrid arch.",
        "ship": "Hybrid vs. CNN",
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{lrrrc}",
        "\\toprule",
        "Comparison & $\\Delta$ top-1 & 95\\% CI & $p_{\\mathrm{Holm}}$ & Signers \\\\",
        "\\midrule",
    ]
    for r in rows:
        k = keys[r["comparison"]]
        f = f"comparison={r['comparison']}"
        d = m.pct(f"cmp.{k}.diff", float(r["diff"]), 1, path, f"diff|{f}", signed=True)
        c = m.ci(
            f"cmp.{k}.ci", float(r["ci_low"]), float(r["ci_high"]), 1, path, f"ci_low,ci_high|{f}"
        )
        p = m.p(f"cmp.{k}.p", float(r["p_holm"]), path, f"p_holm|{f}")
        m.sig(f"cmp.{k}.sig", float(r["p_holm"]), path, f"p_holm|{f}")
        pos = m.num(
            f"cmp.{k}.signers", int(r["signers_positive"]), 0, path, f"signers_positive|{f}"
        )
        m.num(f"cmp.{k}.nsigners", int(r["signers"]), 0, path, f"signers|{f}")
        lines.append(f"{short[k]} & {d} & {c} & {p} & {pos}/{r['signers']} \\\\")
        if k == "tf":  # standalone v1 comparisons above, the cumulative v2 steps below
            lines.append("\\midrule")
    m.num("cmp.n", int(rows[0]["n_sequences"]), 0, path, "n_sequences|comparison=v1: GRU vs CNN")
    m.num("cmp.family", len(rows), 0, path, "derived: number of rows")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Paired comparisons on the test split (percentage points of top-1; each sequence's "
        "score is its correctness averaged over the three seeds). The + rows add one change to the "
        "previous model: v2 features to the v1 CNN, then the v2 recipe, then the hybrid architecture; "
        "the last row compares the hybrid with the v1 CNN. Intervals: paired bootstrap over "
        "sequences; $p$: sign-flip permutation test, Holm-adjusted over the six rows. Signers: test "
        "signers (of four) on whom the first model is better; sequences of one signer are not "
        "independent, so this column is the more conservative evidence.}",
        "\\label{tab:comparisons}",
        "\\end{table}",
    ]
    write_table("comparisons", "\n".join(lines))


# --------------------------------------------------------------------------------- efficiency


def efficiency(m: Macros) -> None:
    quant_path = ROOT / "results" / "tables" / "test_quant_summary.csv"
    quant = read_csv(quant_path)
    bench_path = ROOT / "results" / "tables" / "bench_summary.csv"
    bench = read_csv(bench_path)
    bench2_path = ROOT / "results" / "bench_v2" / "tables" / "bench_summary.csv"
    bench2 = read_csv(bench2_path)
    names = {
        "fp32": "fp32",
        "fp16": "fp16",
        "int8_dynamic": "int8 dyn.",
        "int8_static": "int8 st.",
        "int8_static_full": "int8 full",
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Variant & Top-1 & $\\Delta$ & MB & p50 (ms) \\\\",
        "\\midrule",
    ]
    for model, label in (("cnn", "CNN"), ("gru", "GRU"), ("transformer", "Transf.")):
        first = True
        for variant in ("fp32", "fp16", "int8_dynamic", "int8_static", "int8_static_full"):
            q = row(quant, model=model, variant=variant)
            f = f"model={model}|variant={variant}"
            key = f"q.{model}.{variant.replace('_', '')}"
            top1 = m.pct(f"{key}.top1", float(q["top1_mean"]), 1, quant_path, f"top1_mean|{f}")
            delta = m.pct(
                f"{key}.delta",
                float(q["delta_vs_fp32_mean"]),
                1,
                quant_path,
                f"delta_vs_fp32_mean|{f}",
                signed=True,
            )
            size = m.num(f"{key}.mb", float(q["size_mb"]), 2, quant_path, f"size_mb|{f}")
            bvariant = "onnx_fp32" if variant == "fp32" else variant
            b = row(bench, experiment=f"{model}_signer_independent", variant=bvariant, threads="1")
            lat = m.num(
                f"{key}.p50",
                float(b["p50_ms"]),
                2,
                bench_path,
                f"p50_ms|experiment={model}_signer_independent|variant={bvariant}|threads=1",
            )
            m.num(
                f"{key}.agree",
                float(q["agreement_mean"]) * 100,
                1,
                quant_path,
                f"agreement_mean|{f}",
                scale=1.0,
            )
            m.items[f"{key}.agree"]["raw"] = float(q["agreement_mean"])
            m.items[f"{key}.agree"]["scale"] = 100.0
            lines.append(
                f"{label if first else ''} & {names[variant]} & {top1} & {delta} & {size} & {lat} \\\\"
            )
            first = False
        lines.append("\\midrule")
    for variant, vname in (("onnx_fp32", "fp32"), ("fp16", "fp16"), ("int8_dynamic", "int8 dyn.")):
        b = row(bench2, experiment="d_hybrid_v2feat_v2recipe", variant=variant, threads="1")
        key = f"q.hybrid.{variant.replace('_', '')}"
        lat = m.num(
            f"{key}.p50",
            float(b["p50_ms"]),
            2,
            bench2_path,
            f"p50_ms|experiment=d_hybrid_v2feat_v2recipe|variant={variant}|threads=1",
        )
        size = m.num(
            f"{key}.mb",
            float(b["size_mb"]),
            2,
            bench2_path,
            f"size_mb|experiment=d_hybrid_v2feat_v2recipe|variant={variant}|threads=1",
        )
        lines.append(
            f"{'Hybrid' if variant == 'onnx_fp32' else ''} & {vname} & -- & -- & {size} & {lat} \\\\"
        )
    for variant in ("torch_fp32",):
        for model in ("cnn", "gru", "transformer"):
            b = row(bench, experiment=f"{model}_signer_independent", variant=variant, threads="1")
            m.num(
                f"q.{model}.torch.p50",
                float(b["p50_ms"]),
                2,
                bench_path,
                f"p50_ms|experiment={model}_signer_independent|variant={variant}|threads=1",
            )
        b = row(bench2, experiment="d_hybrid_v2feat_v2recipe", variant=variant, threads="1")
        m.num(
            "q.hybrid.torch.p50",
            float(b["p50_ms"]),
            2,
            bench2_path,
            "p50_ms|experiment=d_hybrid_v2feat_v2recipe|variant=torch_fp32|threads=1",
        )
    for threads in ("4",):
        b = row(bench2, experiment="d_hybrid_v2feat_v2recipe", variant="fp16", threads=threads)
        m.num(
            f"q.hybrid.fp16.p50t{threads}",
            float(b["p50_ms"]),
            2,
            bench2_path,
            f"p50_ms|experiment=d_hybrid_v2feat_v2recipe|variant=fp16|threads={threads}",
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Post-training quantization of the v1 models (test top-1 \\%, mean over seeds; "
        "$\\Delta$ against fp32 in points) and CPU latency of one call in Python onnxruntime on the "
        "i3-1115G4 (1 intra-op thread, median p50 over three rounds, seed 0). dyn.: dynamic; st.: "
        "static with the feature block kept in float; full: static with the input and feature block "
        "quantized too. The hybrid has no int8 test evaluation (only latency and size).}",
        "\\label{tab:quant}",
        "\\end{table}",
    ]
    write_table("quant", "\n".join(lines))
    # Browser: the shipped fp16 file, wasm with 1/2/4 threads and WebGPU (results/perf/classifier_bench.json).
    cb_path = ROOT / "results" / "perf" / "classifier_bench.json"
    cb = read_json(cb_path)
    for r in cb["results"]:
        key = f"b.{r['ep']}{r['threads']}"
        sel = f"results[ep={r['ep']};threads={r['threads']}]"
        m.num(f"{key}.p50", r["p50"], 1, cb_path, f"{sel}.p50")
        m.num(f"{key}.p95", r["p95"], 1, cb_path, f"{sel}.p95")
    m.word("b.chrome", cb["browser"].split(".")[0], cb_path, "derived: major version of browser")
    # Tracking: two 60 s runs per HUD setting, fps = frames / duration.
    perf = ROOT / "results" / "perf"
    for hud, files in (
        ("on", ["tracking_duet_final_gpu.json", "tracking_duet_final_gpu.r2.json"]),
        ("off", ["tracking_duet_final_gpu_nohud.json", "tracking_duet_final_gpu_nohud.r2.json"]),
    ):
        reports = [read_json(perf / f)["report"] for f in files]
        # end_to_end_fps = frames processed / duration, as the benchmark script prints it.
        fps = [r["frames"]["end_to_end_fps"] for r in reports]
        for i, (f, value) in enumerate(zip(files, fps, strict=True), 1):
            m.num(f"t.hud{hud}.fps{i}", value, 1, perf / f, "report.frames.end_to_end_fps")
        m.num(
            f"t.hud{hud}.fps",
            float(np.mean(fps)),
            1,
            f"results/perf/{files[0]} + {files[1]}",
            "derived: mean over the two runs of report.frames.end_to_end_fps",
        )
        for i, (f, r) in enumerate(zip(files, reports, strict=True), 1):
            s = r["summary"]
            m.num(
                f"t.hud{hud}.hand{i}",
                s["hand_landmarker_ms"]["p50"],
                1,
                perf / f,
                "report.summary.hand_landmarker_ms.p50",
            )
            m.num(
                f"t.hud{hud}.cls{i}",
                s["classifier_ms"]["p50"],
                1,
                perf / f,
                "report.summary.classifier_ms.p50",
            )
            m.num(
                f"t.hud{hud}.cls95{i}",
                s["classifier_ms"]["p95"],
                1,
                perf / f,
                "report.summary.classifier_ms.p95",
            )
            m.num(
                f"t.hud{hud}.e2e{i}",
                s["segment_end_to_end_ms"]["p50"],
                0,
                perf / f,
                "report.summary.segment_end_to_end_ms.p50",
            )
    env_path = ROOT / "results" / "bench" / "bench_env.json"
    env = read_json(env_path)["run_info"]
    m.num("hw.cores", env["cpu_physical_cores"], 0, env_path, "run_info.cpu_physical_cores")
    m.num("hw.threads", env["cpu_logical_cores"], 0, env_path, "run_info.cpu_logical_cores")
    m.num(
        "hw.ramgib",
        env["ram_total_bytes"] / 2**30,
        1,
        env_path,
        "run_info.ram_total_bytes",
        scale=1.0,
    )
    m.items["hw.ramgib"]["raw"] = float(env["ram_total_bytes"])
    m.items["hw.ramgib"]["scale"] = 1 / 2**30
    first = read_json(perf / "tracking_duet_final_gpu.json")["report"]
    m.num(
        "t.duration",
        first["duration_s"],
        0,
        perf / "tracking_duet_final_gpu.json",
        "report.duration_s",
    )
    m.num(
        "budget.ms",
        60,
        0,
        ROOT / "results" / "model_selection.md",
        "derived: the selection rule's latency budget (p95 < 60 ms), fixed before measuring",
    )
    cpu_path = perf / "tracking_cpu.json"
    cpu = read_json(cpu_path)["report"]
    m.num("t.cpu.fps", cpu["frames"]["end_to_end_fps"], 1, cpu_path, "report.frames.end_to_end_fps")
    m.num(
        "t.cpu.hand",
        cpu["summary"]["hand_landmarker_ms"]["p50"],
        1,
        cpu_path,
        "report.summary.hand_landmarker_ms.p50",
    )


# ------------------------------------------------------------------------------------- calibration


def calibration(m: Macros) -> None:
    path = ROOT / "results" / "test_analysis" / "calibration.json"
    c = read_json(path)
    meta_path = ROOT / "web" / "public" / "model" / "meta.json"
    meta = read_json(meta_path)["calibration"]
    m.num("cal.T", c["temperature_from_val"], 3, path, "temperature_from_val")
    m.pct("cal.thr", c["threshold_from_val"], 1, path, "threshold_from_val")
    m.pct("cal.test.top1", c["test_top1"], 1, path, "test_top1")
    m.pct("cal.test.top3", c["test_top3"], 1, path, "test_top3")
    m.num("cal.test.ece0", c["test_ece_before"], 3, path, "test_ece_before")
    m.num("cal.test.ece1", c["test_ece_after"], 3, path, "test_ece_after")
    m.num("cal.test.nll0", c["test_nll_before"], 3, path, "test_nll_before")
    m.num("cal.test.nll1", c["test_nll_after"], 3, path, "test_nll_after")
    m.pct("cal.test.cov", c["test_coverage_at_threshold"], 1, path, "test_coverage_at_threshold")
    m.pct(
        "cal.test.sel",
        c["test_selective_accuracy_at_threshold"],
        1,
        path,
        "test_selective_accuracy_at_threshold",
    )
    m.num("cal.bins", c["ece_bins"], 0, path, "ece_bins")
    m.num("cal.val.ece0", meta["ece_before"], 3, meta_path, "calibration.ece_before")
    m.num("cal.val.ece1", meta["ece_after"], 3, meta_path, "calibration.ece_after")
    m.pct("cal.val.cov", meta["coverage"], 1, meta_path, "calibration.coverage")
    m.pct("cal.val.sel", meta["selective_accuracy"], 1, meta_path, "calibration.selective_accuracy")
    m.num("cal.test.n", c["n_sequences"], 0, path, "n_sequences")
    # What a user waits after the hands stop: the segmenter's end delay plus the measured
    # end-of-segment-to-word time (first HUD-on run).
    seg = read_json(ROOT / "features_spec.json")["live"]["segmenter"]
    tracking = ROOT / "results" / "perf" / "tracking_duet_final_gpu.json"
    e2e = read_json(tracking)["report"]["summary"]["segment_end_to_end_ms"]["p50"]
    m.num(
        "t.userlo",
        seg["hand_lost_ms"] + e2e,
        0,
        f"features_spec.json + {rel(tracking)}",
        "derived: live.segmenter.hand_lost_ms + segment_end_to_end_ms p50",
    )
    m.num(
        "t.userhi",
        seg["still_ms"] + e2e,
        0,
        f"features_spec.json + {rel(tracking)}",
        "derived: live.segmenter.still_ms + segment_end_to_end_ms p50",
    )
    m.num(
        "budget.fps",
        15,
        0,
        ROOT / "README.md",
        "derived: the tracking target (at least 15 fps) stated in the README's Performance section",
    )
    m.num("cal.test.net", c["test_net_score_at_threshold"], 2, path, "test_net_score_at_threshold")
    m.ci("cal.test.ece0ci", *c["test_ece_before_ci"], 3, path, "test_ece_before_ci.0,1", scale=1.0)
    m.ci("cal.test.ece1ci", *c["test_ece_after_ci"], 3, path, "test_ece_after_ci.0,1", scale=1.0)
    m.pct(
        "cal.test.rejwrong",
        c["test_wrong_rejected_at_threshold"],
        0,
        path,
        "test_wrong_rejected_at_threshold",
    )
    m.pct(
        "cal.test.rejright",
        c["test_correct_rejected_at_threshold"],
        0,
        path,
        "test_correct_rejected_at_threshold",
    )


# --------------------------------------------------------------------------------------- live gap


def live_clips(m: Macros) -> None:
    """Metadata of the live clips (no pixels), snapshotted from the git-ignored manifest."""
    manifest_path = ROOT / "data" / "live_clips_v2" / "clips.json"
    snapshot = DATA / "live_clips_v2.json"
    if manifest_path.exists():
        clips = read_json(manifest_path)["clips"]
        keep = (
            "label",
            "take",
            "sha256",
            "source_fps",
            "source_frames",
            "source_size",
            "y4m_size",
            "duplicate_of",
        )
        snap = {"source": rel(manifest_path), "clips": [{k: c[k] for k in keep} for c in clips]}
        snapshot.write_text(json.dumps(snap, indent=1) + "\n", encoding="utf-8")
    clips = read_json(snapshot)["clips"]
    if len({tuple(c["source_size"]) for c in clips}) != 1 or any(c["duplicate_of"] for c in clips):
        raise ValueError("live clips differ in size or contain duplicates; revise the text")
    durations = [c["source_frames"] / c["source_fps"] for c in clips]
    m.num(
        "lg.duration",
        float(np.median(durations)),
        1,
        snapshot,
        "derived: median of source_frames / source_fps",
    )
    m.num(
        "lg.fps",
        float(np.median([c["source_fps"] for c in clips])),
        0,
        snapshot,
        "derived: median source_fps",
    )
    m.num(
        "lg.width",
        clips[0]["source_size"][0],
        0,
        snapshot,
        "derived: source_size[0] (same for all)",
    )
    m.num(
        "lg.height",
        clips[0]["source_size"][1],
        0,
        snapshot,
        "derived: source_size[1] (same for all)",
    )
    m.num("lg.ywidth", clips[0]["y4m_size"][0], 0, snapshot, "derived: y4m_size[0] (same for all)")
    m.num("lg.yheight", clips[0]["y4m_size"][1], 0, snapshot, "derived: y4m_size[1] (same for all)")


def live_gap(m: Macros) -> None:
    path = ROOT / "results" / "live_gap_v2" / "summary.json"
    s = read_json(path)
    m.num("lg.clips", s["clips"], 0, path, "clips")
    m.num("lg.signs", s["signs"], 0, path, "signs")
    rows = {(r["pipeline"], r["config"]): r for r in s["rows"]}
    cmp = {(c["pipeline"], c["metric"], c["config"]): c for c in s["comparisons"]}
    pipe = {"python": "py", "web": "web"}
    for (pipeline, config), r in rows.items():
        key = f"lg.{pipe[pipeline]}.{config.replace('_', '')}"
        sel = f"rows[pipeline={pipeline};config={config}]"
        for metric in ("top1", "top3"):
            m.pct(f"{key}.{metric}", r[f"clip_{metric}"], 0, path, f"{sel}.clip_{metric}")
            m.ci(
                f"{key}.{metric}ci",
                r[f"clip_{metric}_ci_low"],
                r[f"clip_{metric}_ci_high"],
                0,
                path,
                f"{sel}.clip_{metric}_ci_low,clip_{metric}_ci_high",
            )
        m.num(f"{key}.runs", r["runs"], 0, path, f"{sel}.runs")
        m.num(f"{key}.fps", r["fps"], 1, path, f"{sel}.fps")
        m.pct(f"{key}.seg", r["segment_top1"], 0, path, f"{sel}.segment_top1")
        m.pct(f"{key}.cov", r["coverage"], 0, path, f"{sel}.coverage")
        m.pct(f"{key}.selacc", r["selective_accuracy"], 0, path, f"{sel}.selective_accuracy")
        m.num(f"{key}.segms", r["segment_ms_median"], 0, path, f"{sel}.segment_ms_median")
        m.num(f"{key}.segn", r["segments_per_run"], 0, path, f"{sel}.segments_per_run")
        m.pct(f"{key}.rejectwrong", r["wrong_rejected"], 0, path, f"{sel}.wrong_rejected")
    for (pipeline, metric, config), c in cmp.items():
        key = f"lg.{pipe[pipeline]}.{config.replace('_', '')}.d{metric}"
        sel = f"comparisons[pipeline={pipeline};metric={metric};config={config}]"
        m.pct(key, c["diff"], 0, path, f"{sel}.diff", signed=True)
        m.ci(f"{key}ci", c["ci_low"], c["ci_high"], 0, path, f"{sel}.ci_low,ci_high")
        m.p(f"{key}p", c["p_holm"], path, f"{sel}.p_holm")
        m.sig(f"{key}sig", c["p_holm"], path, f"{sel}.p_holm")
        m.num(f"{key}better", c["clips_better"], 0, path, f"{sel}.clips_better")
        m.num(f"{key}worse", c["clips_worse"], 0, path, f"{sel}.clips_worse")
    m.num(
        "lg.family",
        sum(1 for c in s["comparisons"] if c["pipeline"] == "web" and c["metric"] == "top1"),
        0,
        path,
        "derived: number of web top-1 comparisons",
    )
    m.num(
        "lg.familypy",
        sum(1 for c in s["comparisons"] if c["pipeline"] == "python" and c["metric"] == "top1"),
        0,
        path,
        "derived: number of python top-1 comparisons",
    )
    # Share of the hybrid's drop (test top-1 -> live with only the model swapped in) that the
    # fixes recover: (c6_final - c1_model) / (test top-1 - c1_model).
    test_top1 = float(
        row(
            read_csv(ROOT / "results" / "tables" / "selection" / "test_fp32_summary.csv"),
            experiment="d_hybrid_v2feat_v2recipe",
        )["test_top1_mean"]
    )
    for pipeline, short in pipe.items():
        if (pipeline, "c6_final") in rows and (pipeline, "c1_model") in rows:
            final, start = (
                rows[(pipeline, "c6_final")]["clip_top1"],
                rows[(pipeline, "c1_model")]["clip_top1"],
            )
            m.pct(
                f"lg.{short}.recover",
                (final - start) / (test_top1 - start),
                0,
                f"{rel(path)} + results/tables/selection/test_fp32_summary.csv",
                "derived: (c6_final - c1_model) / (hybrid test top-1 - c1_model)",
            )
    # Sign-level sensitivity analysis and run-to-run variation.
    for (pipeline, metric, config), c in cmp.items():
        if metric != "top1":
            continue
        key = f"lg.{pipe[pipeline]}.{config.replace('_', '')}.d{metric}"
        sel = f"comparisons[pipeline={pipeline};metric={metric};config={config}]"
        m.p(f"{key}psign", c["p_holm_sign"], path, f"{sel}.p_holm_sign")
        m.sig(f"{key}sigsign", c["p_holm_sign"], path, f"{sel}.p_holm_sign")
        m.ci(
            f"{key}cisign",
            c["ci_low_sign"],
            c["ci_high_sign"],
            0,
            path,
            f"{sel}.ci_low_sign,ci_high_sign",
        )
        m.p(f"{key}praw", c["p"], path, f"{sel}.p")
    for (pipeline, config), r in rows.items():
        key = f"lg.{pipe[pipeline]}.{config.replace('_', '')}"
        sel = f"rows[pipeline={pipeline};config={config}]"
        m.num(f"{key}.noseg", r["no_segment_clips"], 1, path, f"{sel}.no_segment_clips")
        m.pct(f"{key}.runsd", r["run_top1_sd"], 0, path, f"{sel}.run_top1_sd")
    m.num("lg.stalled", s["stalled_records"], 0, path, "stalled_records")
    m.num("lg.browserrecords", s["browser_records"], 0, path, "browser_records")
    for d in s["pipeline_diffs"]:
        key = f"lg.pd.{d['config'].replace('_', '')}"
        sel = f"pipeline_diffs[config={d['config']}]"
        m.pct(f"{key}.diff", d["diff"], 0, path, f"{sel}.diff", signed=True)
        m.ci(
            f"{key}.ci",
            d["ci_low_sign"],
            d["ci_high_sign"],
            0,
            path,
            f"{sel}.ci_low_sign,ci_high_sign",
        )
    for c in s["live_calibration"]:
        key = f"lg.cal.{pipe[c['pipeline']]}" + (
            "" if c["config"] == "c6_final" else f".{c['config'].replace('_', '')}"
        )
        sel = f"live_calibration[pipeline={c['pipeline']};config={c['config']}]"
        m.num(f"{key}.ece", c["ece"], 3, path, f"{sel}.ece")
        m.pct(f"{key}.cov", c["coverage"], 0, path, f"{sel}.coverage")
        m.pct(f"{key}.sel", c["selective_accuracy"], 0, path, f"{sel}.selective_accuracy")
        m.num(f"{key}.net", c["net_score"], 2, path, f"{sel}.net_score")
        m.pct(f"{key}.rejright", c["correct_rejected"], 0, path, f"{sel}.correct_rejected")
        m.pct(f"{key}.rejwrong", c["wrong_rejected"], 0, path, f"{sel}.wrong_rejected")
    # Signs that every take fails in both pipelines, and confusions shared by both pipelines.
    per = {(p["pipeline"], p["sign"]): p for p in s["per_sign"]}
    signs = sorted({p["sign"] for p in s["per_sign"]})
    both = [
        x
        for x in signs
        if ("python", x) in per
        and ("web", x) in per
        and per[("python", x)]["top1"] == 0
        and per[("web", x)]["top1"] == 0
    ]
    m.num(
        "lg.failboth",
        len(both),
        0,
        path,
        "derived: signs with clip top-1 0 in both pipelines (c6_final)",
    )
    top_guess = {}
    for c in s["confusions"]:
        top_guess.setdefault((c["pipeline"], c["sign"]), c["predicted"])
    shared = [
        x
        for x in signs
        if top_guess.get(("python", x))
        and top_guess.get(("python", x)) == top_guess.get(("web", x))
        and top_guess[("python", x)] != "(no segment)"
    ]
    m.num(
        "lg.sharedconf",
        len(shared),
        0,
        path,
        "derived: signs whose most frequent wrong guess is the same in both pipelines",
    )
    m.word(
        "lg.sharedconflist",
        ", ".join(f"{tex_escape(x)}$\\to${tex_escape(top_guess[('python', x)])}" for x in shared),
        path,
        "derived: those signs and guesses",
    )

    # Main table: cumulative steps and leave-one-out rows, both pipelines.
    def cell(pipeline: str, config: str, metric: str) -> str:
        key = f"lg.{pipe[pipeline]}.{config.replace('_', '')}"
        if (pipeline, config) not in rows:
            return "--"
        value = f"\\V{{{key}.{metric}}}"
        c = cmp.get((pipeline, metric, config))
        mark = "$^{*}$" if c and c["p_holm"] < ALPHA else ""
        return value + mark

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Python} & \\multicolumn{2}{c}{Browser} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Config & top-1 & top-3 & top-1 & top-3 \\\\",
        "\\midrule",
    ]
    for config in ["c0_head_build", *CUMULATIVE]:
        if not any((p, config) in rows for p in pipe):
            continue
        lines.append(
            f"{STEP_NAMES[config]} & {cell('python', config, 'top1')} & {cell('python', config, 'top3')} & "
            f"{cell('web', config, 'top1')} & {cell('web', config, 'top3')} \\\\"
        )
    lines.append("\\midrule")
    for config in LEAVE_ONE_OUT:
        if not any((p, config) in rows for p in pipe):
            continue
        lines.append(
            f"{STEP_NAMES[config]} & {cell('python', config, 'top1')} & {cell('python', config, 'top3')} & "
            f"{cell('web', config, 'top1')} & {cell('web', config, 'top3')} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Live-gap ablation on the \\V{lg.clips} webcam clips (\\V{lg.signs} signs $\\times$ 2 "
        "takes): clip-level top-1/top-3 (\\%) for Python with MediaPipe Holistic and for the browser "
        "with MediaPipe Tasks. Upper block: one fix added per row; lower block: one "
        "fix of the final pipeline undone. Browser rows are means over three real-time runs. "
        "$^{*}$: the change against the row it modifies is significant after Holm correction "
        "within the pipeline and metric (clip as unit, as pre-specified). Top-1 intervals, raw and "
        "sign-level p-values are in Table~\\ref{tab:livegapci}.}",
        "\\label{tab:livegap}",
        "\\end{table}",
    ]
    write_table("livegap", "\n".join(lines))
    # Appendix: intervals and comparisons.
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{llrrrrrrl}",
        "\\toprule",
        "Pipeline & Config & Runs & top-1 [95\\% CI] & $\\Delta$ top-1 [95\\% CI] & "
        "$p$ & $p_{\\mathrm{Holm}}$ & $p_{\\mathrm{Holm}}$, sign & Better/worse \\\\",
        "\\midrule",
    ]
    for pipeline, title in (("python", "Python"), ("web", "Browser")):
        for config in ["c0_head_build", *CUMULATIVE, *LEAVE_ONE_OUT]:
            if (pipeline, config) not in rows:
                continue
            key = f"lg.{pipe[pipeline]}.{config.replace('_', '')}"
            c = cmp.get((pipeline, "top1", config))
            delta = (
                (
                    f"\\V{{{key}.dtop1}} \\V{{{key}.dtop1ci}} & \\V{{{key}.dtop1praw}} & \\V{{{key}.dtop1p}} & "
                    f"\\V{{{key}.dtop1psign}} & \\V{{{key}.dtop1better}}/\\V{{{key}.dtop1worse}}"
                )
                if c
                else "-- & -- & -- & -- & --"
            )
            lines.append(
                f"{title} & {STEP_NAMES[config]} & \\V{{{key}.runs}} & \\V{{{key}.top1}} \\V{{{key}.top1ci}} & "
                f"{delta} \\\\"
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += [
        "\\end{tabular}",
        "\\caption{Live-gap v2 in full: clip-level accuracy (\\%) with percentile bootstrap intervals "
        "over the \\V{lg.clips} clips, and each config against the config it changes: paired "
        "bootstrap interval of the difference (points, not adjusted for multiplicity), the sign-flip "
        "$p$ before and after Holm adjustment within the pipeline (the pre-specified analysis), the "
        "Holm-adjusted $p$ with the two takes of a sign as one unit (a sensitivity analysis), and the "
        "number of clips that got better/worse. Runs: browser runs averaged per configuration.}",
        "\\label{tab:livegapci}",
        "\\end{table*}",
    ]
    write_table("livegap_ci", "\n".join(lines))
    # Per sign.
    per = {(p["pipeline"], p["sign"]): p for p in s["per_sign"]}
    signs = sorted({p["sign"] for p in s["per_sign"]})
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        # The prediction lists wrap (ragged right, from the array package).
        "\\begin{tabular}{lrrr>{\\raggedright\\arraybackslash}p{5.5cm}>{\\raggedright\\arraybackslash}p{5.5cm}}",
        "\\toprule",
        "Sign & Python & Browser & Test & Python predictions & Browser predictions \\\\",
        "\\midrule",
    ]
    for sign in signs:
        py, web = per.get(("python", sign)), per.get(("web", sign))

        def acc(x: dict[str, Any] | None) -> str:
            return "--" if x is None else f"{x['top1'] * 100:.0f}"

        def preds(x: dict[str, Any] | None) -> str:
            return (
                "--" if x is None else tex_escape(x["predictions"].replace("(none)", "no segment"))
            )

        test_acc = py["test_top1"] if py else (web["test_top1"] if web else float("nan"))
        lines.append(
            f"{tex_escape(sign)} & {acc(py)} & {acc(web)} & "
            f"{'--' if math.isnan(test_acc) else f'{test_acc * 100:.0f}'} & "
            f"{preds(py)} & {preds(web)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Per sign, final pipeline: live clip top-1 (\\%) over the two takes in Python and in "
        "the browser (and its runs), the shipped model's top-1 (\\%, Test) on the same sign in the test split, and the "
        "clip-level predictions with counts (``no segment'': no classified segment in the clip). "
        "Generated from \\cfg{results/live\\_gap\\_v2/summary.json}.}",
        "\\label{tab:persign}",
        "\\end{table*}",
    ]
    write_table("persign", "\n".join(lines))
    for pipeline in pipe:
        top = [c for c in s["confusions"] if c["pipeline"] == pipeline][:5]
        m.word(
            f"lg.{pipe[pipeline]}.confusions",
            "; ".join(
                f"{tex_escape(c['sign'])}$\\to${tex_escape(c['predicted'])} ({c['count']}/{c['out_of']})"
                for c in top
            ),
            path,
            "derived: first five confusions of the pipeline",
        )


def pilot(m: Macros) -> None:
    path = ROOT / "results" / "live_gap.csv"
    rows = read_csv(path)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Config & Python & Browser \\\\",
        "\\midrule",
    ]
    for config in ["c0_head_build", *CUMULATIVE]:
        cells = []
        for pipeline, short in (("python", "py"), ("web", "web")):
            r = [x for x in rows if x["pipeline"] == pipeline and x["config"] == config]
            if not r:
                cells.append("--")
                continue
            cells.append(
                m.num(
                    f"pilot.{short}.{config.replace('_', '')}",
                    float(r[0]["clean_clip_top1"]) * 6,
                    0,
                    path,
                    f"clean_clip_top1|pipeline={pipeline}|config={config}",
                    scale=1.0,
                )
            )
            m.items[f"pilot.{short}.{config.replace('_', '')}"]["raw"] = float(
                r[0]["clean_clip_top1"]
            )
            m.items[f"pilot.{short}.{config.replace('_', '')}"]["scale"] = 6.0
        lines.append(f"{STEP_NAMES[config]} & {cells[0]}/6 & {cells[1]}/6 \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Pilot: clips correct (clip-level top-1) among the six pilot clips that are not "
        "byte-identical duplicates; the two duplicate pairs are excluded. Browser rows are means "
        "over one to three runs (Appendix~\\ref{app:pilot}).}",
        "\\label{tab:pilot}",
        "\\end{table}",
    ]
    write_table("pilot", "\n".join(lines))


def parity(m: Macros) -> None:
    path = ROOT / "web" / "tests" / "golden" / "live_parity.json"
    g = read_json(path)
    m.num("par.cases", len(g["dataset_cases"]), 0, path, "derived: len(dataset_cases)")
    m.put("par.tol", g["tolerance"], "$10^{-4}$", path, "tolerance")
    spec_path = ROOT / "features_spec.json"
    seg = read_json(spec_path)["live"]["segmenter"]
    m.num("seg.start", seg["start_energy"], 2, spec_path, "live.segmenter.start_energy")
    m.num("seg.still", seg["still_energy"], 2, spec_path, "live.segmenter.still_energy")
    m.num("seg.stillms", seg["still_ms"], 0, spec_path, "live.segmenter.still_ms")
    m.num("seg.lostms", seg["hand_lost_ms"], 0, spec_path, "live.segmenter.hand_lost_ms")
    m.num("seg.maxs", seg["max_ms"] / 1000, 0, spec_path, "live.segmenter.max_ms", scale=1.0)
    m.items["seg.maxs"]["raw"] = float(seg["max_ms"])
    m.items["seg.maxs"]["scale"] = 0.001
    aspect = read_json(spec_path)["aspect"]
    m.num(
        "asp.train",
        aspect["train_width_over_height"],
        3,
        spec_path,
        "aspect.train_width_over_height",
    )
    m.num("seg.minms", seg["min_ms"], 0, spec_path, "live.segmenter.min_ms")
    m.num("seg.gapms", seg["max_gap_ms"], 0, spec_path, "live.segmenter.max_gap_ms")
    # Constants stated in the derivation note of live.segmenter (features_spec.json).
    derived = seg["derived"]
    for key, needle, value in (
        ("seg.assumedfps", "30 fps", 30),
        ("seg.startpct", "P10", 10),
        ("seg.stillpct", "5%", 5),
    ):
        if needle not in derived:
            raise ValueError(f"features_spec.json live.segmenter.derived no longer says {needle!r}")
        m.num(key, value, 0, spec_path, f"derived: live.segmenter.derived states {needle}")
    # The first demo's presence segmenter and decision rule (as the current app still offers them).
    tracking = ROOT / "results" / "perf" / "tracking_duet_final_gpu.json"
    settings = read_json(tracking)["report"]["context"]["settings"]
    m.num("pres.start", settings["startFrames"], 0, tracking, "report.context.settings.startFrames")
    m.num("pres.end", settings["endFrames"], 0, tracking, "report.context.settings.endFrames")
    m.num("pres.max", settings["maxDurationS"], 0, tracking, "report.context.settings.maxDurationS")
    meta_path = ROOT / "web" / "public" / "model" / "meta.json"
    m.pct(
        "cal.targetprec",
        read_json(meta_path)["threshold"]["target_precision"],
        0,
        meta_path,
        "threshold.target_precision",
    )


def hparams(m: Macros) -> None:
    def cfg(name: str) -> dict[str, Any]:
        import yaml  # local import: only the appendix table needs it

        return dict(yaml.safe_load((ROOT / name).read_text(encoding="utf-8")))

    v1 = {m: cfg(f"experiments/train_{m}.yaml") for m in ("gru", "cnn", "transformer")}
    v2 = cfg("results/runs/d_hybrid_v2feat_v2recipe/seed_0/config.yaml")
    v1_path, v2_path = (
        ROOT / "experiments" / "train_cnn.yaml",
        ROOT / "results" / "runs" / "d_hybrid_v2feat_v2recipe" / "seed_0" / "config.yaml",
    )
    m.num("hp.v1.epochs", v1["cnn"]["epochs"], 0, v1_path, "yaml: epochs")
    m.num("hp.v2.epochs", v2["epochs"], 0, v2_path, "yaml: epochs")
    m.num("hp.v2.ema", v2["ema_decay"], 3, v2_path, "yaml: ema_decay")
    val_path = ROOT / "results" / "tables" / "selection" / "val_summary.csv"
    m.num(
        "hp.seeds",
        int(row(read_csv(val_path), experiment="d_hybrid_v2feat_v2recipe")["n_seeds"]),
        0,
        val_path,
        "n_seeds|experiment=d_hybrid_v2feat_v2recipe",
    )

    def fmt_model(c: dict[str, Any]) -> str:
        return ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in c["model"].items() if k != "name")

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{lp{12.5cm}}",
        "\\toprule",
        "Setting & Value (from the run configs) \\\\",
        "\\midrule",
    ]
    for name, c in (
        ("GRU (v1)", v1["gru"]),
        ("CNN (v1)", v1["cnn"]),
        ("Transformer (v1)", v1["transformer"]),
        ("Hybrid (v2)", v2),
    ):
        lines.append(f"{name} & {tex_escape(fmt_model(c))} \\\\")
    for name, c in (("v1 recipe", v1["cnn"]), ("v2 recipe", v2)):
        keys = [
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "warmup_fraction",
            "label_smoothing",
            "grad_clip",
            "patience",
        ]
        text = ", ".join(f"{k.replace('_', ' ')} {c[k]}" for k in keys)
        if "ema_decay" in c:
            text += f", EMA decay {c['ema_decay']}"
        aug = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in c["augment"].items())
        lines.append(f"{name} & {tex_escape(text)}; augmentation: {tex_escape(aug)} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Hyperparameters, read from \\cfg{experiments/train\\_*.yaml} (v1) and the hybrid run's "
        "\\cfg{config.yaml} (v2). Every signer-independent configuration was trained with seeds 0, 1 and 2.}",
        "\\label{tab:hparams}",
        "\\end{table*}",
    ]
    write_table("hparams", "\n".join(lines))


def mirroring(m: Macros) -> None:
    """Share of each signer's sequences that dominant-hand mirroring flipped (preprocessing
    metadata only), snapshotted from the git-ignored manifest."""
    manifest = ROOT / "data" / "processed_v2" / "manifest.csv"
    split_path = (
        ROOT
        / "results"
        / "runs"
        / "d_hybrid_v2feat_v2recipe"
        / "seed_0"
        / "splits_signer_independent.json"
    )
    snapshot = DATA / "mirrored_by_signer.json"
    if manifest.exists():
        part = {str(p): s for s, ps in read_json(split_path)["participants"].items() for p in ps}
        counts: dict[str, list[int]] = {}
        for r in read_csv(manifest):
            if r["excluded"] == "True":
                continue
            c = counts.setdefault(r["participant_id"], [0, 0])
            c[0] += r["mirrored"] == "True"
            c[1] += 1
        rows = [
            {
                "participant_id": p,
                "split": part[p],
                "mirrored_fraction": c[0] / c[1],
                "sequences": c[1],
            }
            for p, c in sorted(counts.items())
        ]
        snapshot.write_text(
            json.dumps({"source": rel(manifest), "signers": rows}, indent=1) + "\n",
            encoding="utf-8",
        )
    rows = read_json(snapshot)["signers"]
    m.num(
        "mir.high",
        sum(r["mirrored_fraction"] > 0.8 for r in rows),
        0,
        snapshot,
        "derived: signers with more than 80% mirrored",
    )
    m.num(
        "mir.low",
        sum(r["mirrored_fraction"] < 0.05 for r in rows),
        0,
        snapshot,
        "derived: signers with under 5% mirrored",
    )
    m.pct(
        "mir.testmax",
        max(r["mirrored_fraction"] for r in rows if r["split"] == "test"),
        1,
        snapshot,
        "derived: largest mirrored fraction among the test signers",
    )


def test_access() -> None:
    """Every logged invocation of the test-split evaluation script."""
    logs = [
        ROOT / "results" / d / "test_access_log.json"
        for d in ("final", "final_v2", "final_selection")
    ]
    entries = [(e, log) for log in logs for e in read_json(log)]
    entries.sort(key=lambda x: x[0]["timestamp_utc"])
    describe = {
        "quant": "quantized v1 variants",
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llp{4.0cm}}",
        "\\toprule",
        "Date (UTC) & Commit & Evaluated on test \\\\",
        "\\midrule",
    ]
    for e, _ in entries:
        kinds = sorted({k.split("/")[0] for k in e["runs_evaluated"]})
        if kinds == ["quant"]:
            what = describe["quant"]
        else:
            names = []
            if any(k.endswith("signer_dependent") for k in kinds):
                names.append("signer-dependent v1")
            if any(k.endswith("signer_independent") for k in kinds):
                names.append("v1 models")
            if any(k.startswith(("b_", "c_", "d_")) for k in kinds):
                names.append("v2 models")
            if any(k.startswith("onnx:") for k in kinds):
                names.append("shipped ONNX file")
            what = ", ".join(names)
        if e["timestamp_utc"] >= "2026-11-16":
            what += " (saved logits for this paper)"
        dirty = "+" if e.get("git_dirty") else ""
        lines.append(f"{e['timestamp_utc'][:10]} & {e['git_commit'][:7]}{dirty} & {what} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Every logged run of the only script that reads the test split "
        "(\\cfg{results/final*/test\\_access\\_log.json}); + marks a working tree with uncommitted "
        "changes. The v2 models were designed after the v1 evaluation in the first row.}",
        "\\label{tab:testaccess}",
        "\\end{table}",
    ]
    write_table("testaccess", "\n".join(lines))


def per_signer(m: Macros) -> None:
    path = ROOT / "results" / "tables" / "selection" / "test_per_participant.csv"
    rows = read_csv(path)
    # The four signer-independent test signers (the file also holds signer-dependent rows).
    signers = sorted(
        {r["participant_id"] for r in rows if r["experiment"] == "d_hybrid_v2feat_v2recipe"}
    )
    shown = [
        ("cnn_signer_independent", "CNN (v1)"),
        ("transformer_signer_independent", "Transformer (v1)"),
        ("gru_signer_independent", "GRU (v1)"),
        ("b_cnn_v2feat_v1recipe", "CNN, v2 features"),
        ("c_cnn_v2feat_v2recipe", "CNN, v2 features + recipe"),
        ("d_hybrid_v2feat_v2recipe", "Hybrid"),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l" + "r" * len(signers) + "}",
        "\\toprule",
        "Model & " + " & ".join(f"S{i + 1}" for i in range(len(signers))) + " \\\\",
        "\\midrule",
    ]
    for exp, name in shown:
        cells = []
        for s in signers:
            r = row(rows, experiment=exp, participant_id=s)
            cells.append(f"{float(r['test_top1_mean']) * 100:.1f}")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    hyb = [
        float(row(rows, experiment="d_hybrid_v2feat_v2recipe", participant_id=s)["test_top1_mean"])
        for s in signers
    ]
    m.pct("sig.d.min", min(hyb), 1, path, "derived: lowest per-signer test top-1 of the hybrid")
    m.pct("sig.d.max", max(hyb), 1, path, "derived: highest per-signer test top-1 of the hybrid")
    seqs = {
        s: row(rows, experiment="d_hybrid_v2feat_v2recipe", participant_id=s)["n_sequences"]
        for s in signers
    }
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Test top-1 (\\%, mean over seeds) for each of the four test signers "
        f"(S1--S4: {', '.join(seqs[s] for s in signers)} sequences).}}",
        "\\label{tab:persigner}",
        "\\end{table}",
    ]
    write_table("persigner", "\n".join(lines))


def power(m: Macros) -> None:
    """The smallest number of clips that must all move in one direction for the first Holm step
    to reach p < 0.05 with the exact sign-flip test (a property of the test, not a result)."""
    path = ROOT / "results" / "live_gap_v2" / "summary.json"
    s = read_json(path)
    for pipeline, short in (("python", "py"), ("web", "web")):
        family = sum(
            1 for c in s["comparisons"] if c["pipeline"] == pipeline and c["metric"] == "top1"
        )
        k = next(k for k in range(1, 60) if 2 * 0.5**k <= ALPHA / family)
        m.num(
            f"mde.{short}.k", k, 0, path, f"derived: smallest k with 2 x 0.5^k <= 0.05 / {family}"
        )
        m.num(
            f"mde.{short}.family",
            family,
            0,
            path,
            "derived: number of top-1 comparisons of the pipeline",
        )
        m.pct(f"mde.{short}.pts", k / s["clips"], 0, path, "derived: k / number of clips")


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    m = Macros()
    dataset(m)
    models(m)
    comparisons(m)
    efficiency(m)
    calibration(m)
    live_clips(m)
    live_gap(m)
    pilot(m)
    parity(m)
    hparams(m)
    mirroring(m)
    test_access()
    per_signer(m)
    power(m)
    m.write()
    print(f"{len(m.items)} macros -> paper/tables/numbers.tex; tables in paper/tables/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
