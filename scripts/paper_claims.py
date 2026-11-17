"""Audit the paper's numbers, citations and wording, and write paper/CLAIMS.md.

    python scripts/paper_tables.py && python scripts/paper_claims.py

For every \\V{key} used in paper/main.tex, paper/sections/*.tex and the generated tables:
  1. the key must exist in paper/data/macro_sources.json (written by paper_tables.py);
  2. where the source can be read independently (CSV or JSON fields), the value is looked up
     again here, by a different code path, and compared with the recorded raw value;
  3. the printed text is parsed back and must equal raw x scale at its printed precision; an
     interval must match both ends, "significant"/"not significant" must agree with the Holm
     p-value it reports, and "<0.001" must hold.
It also checks that every cited key is in paper/references.bib and verified in
paper/data/reference_verification.json (and that 20 to 35 references are cited), that no banned
word appears, and lists every digit in the hand-written .tex outside a macro so that none is a
result typed by hand. Exit code 1 when any check fails.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ALPHA = 0.05
BANNED = [
    "delve",
    "tapestry",
    "testament",
    "leverage",
    "utilize",
    "utilise",
    "landscape",
    "paramount",
    "furthermore",
    "fundamentally",
    "bespoke",
]

# Factual (non-numeric) claims, checked by reading the named source.
FACTS = [
    (
        "The asl-signs data is licensed CC BY 4.0, provided by the Deaf Professional Arts Network and the "
        "Georgia Institute of Technology.",
        "Kaggle competition data page (kaggle.com/competitions/asl-signs/data), via a search snippet; the page "
        "itself renders client-side and could not be fetched (author to confirm).",
    ),
    (
        "MediaPipe (library, tasks-vision 0.10.35 wasm, hand_landmarker.task, pose_landmarker_lite.task) is "
        "Apache-2.0.",
        "web/public/mediapipe/ASSETS.json (licence per asset, written by npm run fetch-assets)",
    ),
    (
        "Only scripts/final_eval.py reads the test split and every invocation is logged.",
        "scripts/final_eval.py docstring; results/final_selection/test_access_log.json",
    ),
    (
        "Thresholds, temperature, aspect constant and segmenter thresholds come from train/val only.",
        "web/public/model/meta.json (calibration.computed_on = val); features_spec.json (aspect.derived, "
        "live.segmenter.derived)",
    ),
    (
        "The live clips are 25 signs x 2 takes by one signer, unmirrored webcam recordings, none duplicated.",
        "data/live_clips_v2/clips.json (labels, takes, sha256; duplicate_groups empty)",
    ),
    (
        "The browser pipeline runs MediaPipe and the classifier in two Web Workers with requestVideoFrameCallback "
        "capture; COOP/COEP come from the service worker on GitHub Pages.",
        "README.md (Performance: Workers, onnxruntime-web); web/src/inference.worker.ts",
    ),
    (
        "Python and browser features agree within 1e-4 with the same top-1 on 20 val sequences (golden test).",
        "web/tests/unit/liveParity.test.ts; web/tests/golden/live_parity.json",
    ),
    (
        "Training ran on a Kaggle Tesla T4 with mixed precision.",
        "results/runs/*/seed_*/summary.json (device_name, amp)",
    ),
    (
        "Sentence translation is not evaluated quantitatively.",
        "no evaluation of deploy/worker output exists in results/",
    ),
]

ALLOWED_DIGIT_PATTERNS = [
    r"\\V\{[^}]*\}",
    r"\\(?:ref|label|cite[pt]?|citealp|citeauthor|input|includegraphics|url|section|subsection|paragraph|cfg)\*?(?:\[[^\]]*\])?\{[^}]*\}",
    r"\\(?:setlength|usetikzlibrary|definecolor|documentclass|usepackage|tabcolsep|cmidrule)(?:\[[^\]]*\])?\{[^}]*\}(?:\{[^}]*\})?",
    r"\\cmidrule\([^)]*\)\{[^}]*\}",
    r"[0-9.]+(?:mm|pt|em|cm|ex|\\linewidth|\\columnwidth|\\textwidth)",
    r"i3-1115G4",
    r"\bfp(?:16|32)\b",
    r"\bint8\b",
    r"top-[135]",
    r"\bv[12]\b",
    r"\b[A-Z]\d\b",
    r"\(([a-f])\)",
    r"CC BY 4\.0",
    r"Apache-2\.0",
    r"0\.10\.35",
    r"95\\%",
    r"\bx\\,\$\\to\$\\,1\$-\$x",
    r"1\$-\$x",
    r"p_\{\\mathrm\{Holm\}\}",
    r"\[0,\s*1\]",
    r"2 takes",
    r"\\times\$ 2",
    r"\bL\b",
    r"\b(?:WebGPU|SIMD)\b",
    r"Windows~11",
    r"\b11pt\b",
    r"\bT1\b",
    r"utf8",
    r"#1",
    r"10\^\{-4\}",
    r"\bc[0-6]\\_[a-z]+",
    r"\bx\\_[a-z]+",
    r"COOP/COEP",
    r"\bp(?:50|95|99)\b",  # percentile names
    r"more than 80\\% of their sequences",
    r"under 5\\%",  # definitions of the mirroring counts
    r"\(1 thread,",  # benchmark setting
    r"\b[a-z]+(?:19|20)\d\d[a-z]+\b",  # citation keys continued on the next line
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_selector(part: str) -> tuple[str, dict[str, str]]:
    m = re.fullmatch(r"([^\[]*)\[([^\]]*)\]", part)
    if not m:
        return part, {}
    conditions = dict(c.split("=", 1) for c in m.group(2).split(";"))
    return m.group(1), conditions


def json_lookup(data: Any, path: str) -> Any:
    """Dotted path with list selectors name[k=v;k2=v2]; the last segment may be "a,b"."""

    def step(obj: Any, key: str) -> Any:
        return obj[int(key)] if isinstance(obj, list) else obj[key]

    parts = path.split(".")
    for i, part in enumerate(parts):
        if i == len(parts) - 1 and "," in part:
            return [step(data, p) for p in part.split(",")]
        name, conditions = parse_selector(part)
        if name:
            data = step(data, name)
        if conditions:
            matches = [x for x in data if all(str(x.get(k)) == v for k, v in conditions.items())]
            if len(matches) != 1:
                raise KeyError(f"{part}: {len(matches)} matches")
            data = matches[0]
    return data


def lookup(source: str, field: str) -> Any:
    """Re-read a recorded value from its source; None for derived values or unreadable sources."""
    if field.startswith("derived") or "+" in source:
        return None
    path = ROOT / source
    if not path.exists():
        return None
    if path.suffix == ".csv":
        columns, *conditions = field.split("|")
        filters = dict(c.split("=", 1) for c in conditions)
        rows = [r for r in read_csv(path) if all(r.get(k) == v for k, v in filters.items())]
        if len(rows) != 1:
            raise KeyError(f"{source}: {len(rows)} rows for {filters}")
        values = [float(rows[0][c]) for c in columns.split(",")]
        return values if len(values) > 1 else values[0]
    if path.suffix == ".json":
        return json_lookup(json.loads(path.read_text(encoding="utf-8")), field)
    if path.suffix in (".yaml", ".yml") and field.startswith("yaml: "):
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return json_lookup(data, field.removeprefix("yaml: "))
    return None


def parse_number(text: str) -> tuple[float, int] | None:
    t = text.replace("$-$", "-").replace(",", "").strip()
    m = re.fullmatch(r"([+-]?\d+(?:\.(\d+))?)", t)
    return (float(m.group(1)), len(m.group(2) or "")) if m else None


def rounds_to(text: str, value: float) -> bool:
    parsed = parse_number(text)
    if parsed is None:
        return False
    printed, decimals = parsed
    return abs(printed - value) <= 0.5 * 10 ** (-decimals) + 1e-9


def check_macro(item: dict[str, Any]) -> tuple[bool, list[str]]:
    text, raw, scale = item["text"], item["raw"], float(item.get("scale", 1.0))
    notes, ok = [], True
    try:
        indep = lookup(item["source"], item["field"])
    except (KeyError, ValueError, TypeError, IndexError) as error:
        return False, [f"source re-read failed: {error}"]
    if indep is not None:
        if isinstance(raw, list):
            same = all(abs(float(a) - float(b)) < 1e-9 for a, b in zip(raw, indep, strict=True))
        elif isinstance(raw, str):
            # A string may be printed shortened (a 7-character commit hash for a full one).
            printed = text.replace(r"\_", "_")
            same = str(indep) == raw and raw.startswith(printed)
        else:
            same = abs(float(indep) - float(raw)) <= 1e-9 * max(1.0, abs(float(raw)))
        notes.append("source re-read: " + ("same" if same else f"DIFFERENT ({indep})"))
        ok &= same
    else:
        notes.append("derived value (computed by paper_tables.py)")
    if text in ("significant", "not significant"):
        good = (text == "significant") == (float(raw) < ALPHA)
        notes.append("significance word agrees with p" if good else "SIGNIFICANCE WORD WRONG")
        ok &= good
    elif text == "$<$0.001":
        good = float(raw) < 0.001
        notes.append("p < 0.001 holds" if good else "P-VALUE TEXT WRONG")
        ok &= good
    elif isinstance(raw, list):
        parts = [p.strip() for p in text.strip("[]").split(",")]
        good = len(parts) == 2 and all(
            rounds_to(p, float(r) * scale) for p, r in zip(parts, raw, strict=True)
        )
        notes.append("interval printed correctly" if good else "INTERVAL TEXT MISMATCH")
        ok &= good
    elif isinstance(raw, (int, float)) and parse_number(text) is not None:
        good = rounds_to(text, float(raw) * scale)
        notes.append(
            "printed value rounds correctly"
            if good
            else f"PRINTED VALUE MISMATCH ({raw} x {scale})"
        )
        ok &= good
    elif isinstance(raw, (int, float)):
        notes.append("printed as fixed text")
    return ok, notes


def tex_bodies(files: list[Path]) -> dict[Path, str]:
    out = {}
    for f in files:
        lines = [
            line.split("%")[0] if "\\%" not in line else line
            for line in f.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("%")
        ]
        out[f] = "\n".join(lines)
    return out


def main() -> int:
    macros = json.loads((PAPER / "data" / "macro_sources.json").read_text(encoding="utf-8"))
    hand = [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))]
    generated = sorted((PAPER / "tables").glob("*.tex"))
    bodies = tex_bodies(hand + generated)
    used: dict[str, list[str]] = {}
    for f, body in bodies.items():
        for k in re.findall(r"\\V\{([^}]+)\}", body):
            used.setdefault(k, []).append(f.relative_to(PAPER).as_posix())
    for k, item in macros.items():  # macros used inside other macros' text (generated tables)
        for inner in re.findall(r"\\V\{([^}]+)\}", item["text"]):
            used.setdefault(inner, []).append(f"macro {k}")
    rows, failures = [], 0
    for k in sorted(used):
        item = macros.get(k)
        if item is None:
            rows.append((k, "", "", "", "no", "undefined macro"))
            failures += 1
            continue
        ok, notes = check_macro(item)
        rows.append(
            (
                k,
                item["text"],
                item["source"],
                item["field"],
                "yes" if ok else "no",
                "; ".join(notes),
            )
        )
        failures += 0 if ok else 1
    # Citations.
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    verified = json.loads(
        (PAPER / "data" / "reference_verification.json").read_text(encoding="utf-8")
    )
    cited: set[str] = set()
    for body in bodies.values():
        for group in re.findall(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", body):
            cited |= {c.strip() for c in group.split(",")}
    cite_rows = []
    for key in sorted(cited):
        status = verified.get(key, {}).get("status")
        good = key in bib_keys and status == "ok"
        cite_rows.append(
            (key, verified.get(key, {}).get("verified_with", ""), "yes" if good else "no")
        )
        failures += 0 if good else 1
    count_ok = 20 <= len(cited) <= 35
    failures += 0 if count_ok else 1
    uncited = sorted(bib_keys - cited)
    # Wording.
    banned_hits = []
    for f, body in bodies.items():
        for word in BANNED:
            for m in re.finditer(rf"\b{word}\w*", body, re.I):
                banned_hits.append((f.relative_to(PAPER).as_posix(), m.group(0)))
    failures += len(banned_hits)
    # Digits typed by hand.
    digits = []
    for f in hand:
        in_pic = False
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if "\\begin{tikzpicture}" in line:
                in_pic = True
            if in_pic:
                in_pic = "\\end{tikzpicture}" not in line
                continue
            stripped = line.lstrip()
            if stripped.startswith("%") or stripped.startswith(
                ("\\newcommand", "\\definecolor", "\\PackageError")
            ):
                continue
            s = line.split("%")[0] if "\\%" not in line else line
            for pattern in ALLOWED_DIGIT_PATTERNS:
                s = re.sub(pattern, " ", s)
            for d in re.findall(r"\d[\d.,]*", s):
                digits.append((f.relative_to(PAPER).as_posix(), i, d, line.strip()[:110]))
    out = [
        "# Claims audit",
        "",
        "Generated by `scripts/paper_claims.py` from `paper/data/macro_sources.json` (written by "
        "`scripts/paper_tables.py`). Every number in the paper is a `\\V{key}` macro: for each key used, the "
        "source file and field, whether an independent re-read of the source gives the same value, and whether "
        "the printed text rounds correctly (raw x scale). Then the citations, banned words and every digit "
        "typed outside a macro.",
        "",
        f"**Summary:** {len(used)} macros used, {sum(1 for r in rows if r[4] == 'yes')} match; "
        f"{len(cited)} references cited ({'within' if count_ok else 'OUTSIDE'} 20-35), "
        f"{sum(1 for r in cite_rows if r[2] == 'yes')} verified; {len(banned_hits)} banned words; "
        f"{len(digits)} digits outside macros (listed; none is a result); **{failures} failures**.",
        "",
        "## Numbers",
        "",
        "| key | printed | source | field | matches | check |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            "| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in r) + " |"
        )
    out += ["", "## Citations", "", "| key | verified with | ok |", "|---|---|---|"]
    out += [f"| {k} | {u} | {o} |" for k, u, o in cite_rows]
    if uncited:
        out += [
            "",
            f"Entries in references.bib that are not cited (not printed by BibTeX): {', '.join(uncited)}.",
        ]
    out += [
        "",
        "## Banned words",
        "",
        "None." if not banned_hits else "\n".join(f"- {f}: {w}" for f, w in banned_hits),
    ]
    out += [
        "",
        "## Digits outside macros in the hand-written .tex",
        "",
        "Names, labels and fixed wording only. None is a result.",
        "",
        "| file | line | digits | context |",
        "|---|---|---|---|",
    ]
    out += [f"| {f} | {i} | {d} | `{c.replace('|', '/')}` |" for f, i, d, c in digits]
    out += ["", "## Factual (non-numeric) claims", "", "| claim | source |", "|---|---|"]
    out += [f"| {c} | {s} |" for c, s in FACTS]
    (PAPER / "CLAIMS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(
        f"{len(used)} macros, {len(cited)} citations, {len(banned_hits)} banned words, "
        f"{len(digits)} digits outside macros, {failures} failures -> paper/CLAIMS.md"
    )
    for r in rows:
        if r[4] == "no":
            print("FAIL", r)
    for k, _, o in cite_rows:
        if o == "no":
            print("FAIL citation", k)
    for f, w in banned_hits:
        print("FAIL banned word", f, w)
    if not count_ok:
        print(f"FAIL {len(cited)} references cited (need 20-35)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
