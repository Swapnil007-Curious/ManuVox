"""The Kaggle notebooks: training input detection (v3_kaggle.py locate), the code-source
fingerprint and archive checks, and what each generated cell contains."""

import importlib.util
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from edgevision.data import citizen
from test_citizen import write_landmarks

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(f"kaggle_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = load("build_kaggle_notebooks")
v3 = load("v3_kaggle")


@pytest.fixture(scope="module")
def extraction(tmp_path_factory):
    """A finished extraction output: manifest.csv, shards/ and COMPLETE."""
    root = tmp_path_factory.mktemp("extraction") / "asl-citizen-landmarks"
    by_file = write_landmarks(root, np.random.default_rng(11))
    (root / "COMPLETE").write_text("2026-11-18 00:00:00 UTC")
    return root, by_file


def place(extraction_root: Path, target: Path, shards: str = "folder") -> Path:
    """A copy of the extraction at target with its shard files laid out as named."""
    target.mkdir(parents=True)
    for path in extraction_root.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)
    files = sorted((extraction_root / "shards").glob("shard_*.npz"))
    if shards == "folder":
        shutil.copytree(extraction_root / "shards", target / "shards")
    elif shards == "nested":
        shutil.copytree(extraction_root / "shards", target / "shards" / "shards")
    elif shards == "flat":
        for path in files:
            shutil.copy2(path, target / path.name)
    elif shards in ("zip", "zip-prefixed"):
        with zipfile.ZipFile(target / "shards.zip", "w") as archive:
            for path in files:
                prefix = "shards/" if shards == "zip-prefixed" else ""
                archive.write(path, prefix + path.name)
    else:
        raise ValueError(shards)
    return target


def locate(input_root: Path, unpack_dir: Path, capsys) -> tuple[dict[str, str], str]:
    """Run locate; (the shell assignments, parsed as bash would, and the report on stderr)."""
    assert v3.main(["locate", "--input", str(input_root), "--unpack-dir", str(unpack_dir)]) == 0
    captured = capsys.readouterr()
    values = {}
    for line in captured.out.splitlines():
        name, _, value = line.partition("=")
        if name == "PREVIOUS_RUNS":
            values[name] = " ".join(shlex.split(value[1:-1]))
        else:
            (values[name],) = shlex.split(value) or [""]
    return values, captured.err


def assert_readable(root: Path, by_file: dict) -> None:
    """The training loader reads every tracked video from the folder locate returned."""
    found = citizen.find_root(root)
    manifest = citizen.read_landmark_manifest(found)
    rows = list(citizen.iter_landmarks(found, manifest))
    assert len(rows) == int((manifest["status"] == "ok").sum()) and len(rows) > len(by_file) // 2
    for row, landmarks in rows:
        assert landmarks.shape == (int(row["n_frames"]), 543, 3)
        if row["video_file"] in by_file:
            left = by_file[row["video_file"]]["left_hand"]
            np.testing.assert_allclose(landmarks[:, 468:489], left, equal_nan=True)


def test_locate_uses_a_dataset_folder_in_place(tmp_path, extraction, capsys):
    root = place(extraction[0], tmp_path / "input" / "asl-citizen-landmarks-private" / "x")
    values, report = locate(tmp_path / "input", tmp_path / "unpack", capsys)
    assert Path(values["LANDMARKS"]) == root
    assert "usable" in report and "2 in shards/" in report
    assert not (tmp_path / "unpack" / v3.GATHERED).exists()


@pytest.mark.parametrize("layout", ["nested", "flat", "zip", "zip-prefixed"])
def test_locate_gathers_shards_stored_elsewhere(tmp_path, extraction, capsys, layout):
    deep = tmp_path / "input" / "datasets" / "someone" / "asl-citizen-landmarks-private"
    place(extraction[0], deep / "asl-citizen-landmarks", shards=layout)
    values, report = locate(tmp_path / "input", tmp_path / "unpack", capsys)
    gathered = tmp_path / "unpack" / v3.GATHERED
    assert Path(values["LANDMARKS"]) == gathered, report
    assert "gathered the shard files" in report
    assert sorted(p.name for p in (gathered / "shards").iterdir()) == [
        "shard_00000.npz",
        "shard_00001.npz",
    ]
    assert_readable(gathered, extraction[1])


@pytest.mark.parametrize("inner_shards", ["folder", "zip"])
def test_locate_unpacks_a_zipped_landmark_folder(tmp_path, extraction, capsys, inner_shards):
    staging = place(extraction[0], tmp_path / "staging" / "asl-citizen-landmarks", inner_shards)
    (tmp_path / "input" / "upload").mkdir(parents=True)
    shutil.make_archive(str(tmp_path / "input" / "upload" / "landmarks"), "zip", staging.parent)
    values, report = locate(tmp_path / "input", tmp_path / "unpack", capsys)
    assert values["LANDMARKS"], report
    assert "zip files holding such a manifest.csv: 1" in report
    assert_readable(Path(values["LANDMARKS"]), extraction[1])


def test_locate_skips_smoke_unfinished_and_incomplete_outputs(tmp_path, extraction, capsys):
    source = extraction[0]
    smoke = place(source, tmp_path / "input" / "smoke" / "asl-citizen-landmarks-smoke")
    (smoke / "SMOKE").write_text("smoke test")
    part = place(source, tmp_path / "input" / "part1" / "asl-citizen-landmarks")
    (part / "COMPLETE").unlink()
    holes = place(source, tmp_path / "input" / "broken" / "asl-citizen-landmarks")
    (holes / "shards" / "shard_00001.npz").unlink()
    values, report = locate(tmp_path / "input", tmp_path / "unpack", capsys)
    assert values["LANDMARKS"] == ""
    assert "smoke-test output" in report
    assert "no COMPLETE marker" in report
    assert "1 of the 2 shards the manifest needs are missing, e.g. shard_00001.npz" in report
    assert "no usable ASL Citizen landmarks" in report


def test_locate_prefers_the_dataset_and_ignores_other_manifests(tmp_path, extraction, capsys):
    place(extraction[0], tmp_path / "input" / "a-notebook-output" / "asl-citizen-landmarks")
    dataset = place(
        extraction[0],
        tmp_path / "input" / "asl-citizen-landmarks-private" / "asl-citizen-landmarks",
    )
    processed = tmp_path / "input" / "old-train-output" / "processed_v2"
    processed.mkdir(parents=True)
    (processed / "manifest.csv").write_text("sequence_id,label,participant_id\n1,cat,2\n")
    values, report = locate(tmp_path / "input", tmp_path / "unpack", capsys)
    assert Path(values["LANDMARKS"]) == dataset
    assert "landmark folders (manifest.csv with the extraction's columns): 2" in report


def test_locate_finds_asl_signs_and_earlier_runs_and_quotes_paths(tmp_path, extraction, capsys):
    competition = tmp_path / "input" / "competitions" / "asl signs"
    (competition / "train_landmarks").mkdir(parents=True)
    (competition / "train.csv").write_text("path,participant_id,sequence_id,sign\n")
    runs = tmp_path / "input" / "train v3 output" / "runs" / v3.RUN_NAME
    runs.mkdir(parents=True)
    place(extraction[0], tmp_path / "input" / "asl-citizen-landmarks-private" / "l")
    values, _ = locate(tmp_path / "input", tmp_path / "unpack", capsys)
    assert Path(values["DATA_ROOT"]) == competition
    assert Path(values["PREVIOUS_RUNS"]) == runs


def test_locate_with_nothing_attached(tmp_path, capsys):
    (tmp_path / "input").mkdir()
    values, report = locate(tmp_path / "input", tmp_path / "unpack", capsys)
    assert values == {"DATA_ROOT": "", "LANDMARKS": "", "PREVIOUS_RUNS": ""}
    assert "none found" in report and "no usable ASL Citizen landmarks" in report


# ---- the code source: fingerprint, archive checks and the cells ----


def test_fingerprint_ignores_line_endings_and_follows_the_sources(tmp_path):
    for name in builder.SOURCE_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    assert builder.source_fingerprint(tmp_path) == builder.source_fingerprint()
    labels = tmp_path / "src/edgevision/data/citizen_labels.py"
    labels.write_bytes(labels.read_bytes().replace(b"\n", b"\r\n"))
    assert builder.source_fingerprint(tmp_path) == builder.source_fingerprint()
    (tmp_path / "scripts/v3_kaggle.py").write_text("changed\n")
    assert builder.source_fingerprint(tmp_path) != builder.source_fingerprint()
    printed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_kaggle_notebooks.py"), "--fingerprint"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    assert printed == builder.source_fingerprint()


@pytest.mark.parametrize(
    ("path", "forbidden"),
    [
        ("src/edgevision/data/citizen.py", False),
        ("web/public/model/model.onnx", False),
        ("kaggle/asl_citizen_extract.py", False),
        ("data/asl_citizen/landmarks/manifest.csv", True),
        ("artifacts/v3/v3_bundle.zip", True),
        ("web/models/v3-citizen/model.onnx", True),
        ("test_clips_v2/bed_1.mp4", True),
        ("results/asl_citizen/predictions/v2_test.npz", True),
        ("kaggle.json", True),
        (".env", True),
        ("notes/credentials.txt", True),
    ],
)
def test_forbidden_archive_paths(path, forbidden):
    assert bool(builder.FORBIDDEN.search(path)) is forbidden


def write_archive(path: Path, changes: dict[str, bytes | None]) -> Path:
    """A code archive from the working tree's notebook sources and notebooks, with changes."""
    files: dict[str, bytes | None] = {
        name: (ROOT / name).read_bytes() for name in builder.SOURCE_FILES
    }
    for name in builder.notebook_paths():
        files[f"kaggle/{name}.ipynb"] = (ROOT / "kaggle" / f"{name}.ipynb").read_bytes()
    files[builder.COMMIT_FILE] = b"0" * 40
    files.update(changes)
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            if data is None:
                continue
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return path


def test_archive_problems(tmp_path):
    assert builder.archive_problems(write_archive(tmp_path / "ok.tar.gz", {}), "0" * 40) == []
    no_commit = write_archive(tmp_path / "a.tar.gz", {builder.COMMIT_FILE: None})
    assert builder.archive_problems(no_commit) == ["no SOURCE_COMMIT file"]
    other = builder.archive_problems(write_archive(tmp_path / "b.tar.gz", {}), "1" * 40)
    assert other and "expected 1111" in other[0]
    data = write_archive(tmp_path / "c.tar.gz", {"data/asl_citizen/landmarks/manifest.csv": b"x"})
    assert any("forbidden files" in problem for problem in builder.archive_problems(data))
    stale = write_archive(tmp_path / "d.tar.gz", {"scripts/v3_kaggle.py": b"changed\n"})
    problems = builder.archive_problems(stale)
    assert len(problems) == len(builder.notebook_paths())
    assert all("was built from sources" in problem for problem in problems)


def test_the_committed_tree_has_no_forbidden_files():
    if shutil.which("git") is None:
        pytest.skip("no git")
    names = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert [name for name in names if builder.FORBIDDEN.search(name)] == []


def cells(name: str) -> dict[str, str]:
    """{cell id: source} of kaggle/<name>.ipynb, in order."""
    notebook = json.loads((ROOT / "kaggle" / f"{name}.ipynb").read_text("utf-8"))
    assert notebook["metadata"]["kaggle"]["isInternetEnabled"] is True
    assert notebook["metadata"]["kaggle"]["isGpuEnabled"] is (name == "train_v3")
    return {c["id"]: "".join(c["source"]) for c in notebook["cells"]}


def cell(name: str) -> str:
    """The main cell; only the smoke test has another one, the diagnostics cell, before it."""
    found = cells(name)
    assert list(found) == (["diagnostics", "main"] if name == "asl_citizen_smoke" else ["main"])
    return found["main"]


def test_the_smoke_test_starts_with_the_diagnostics_cell():
    source = cells("asl_citizen_smoke")["diagnostics"]
    assert source.startswith("%%bash\n") and source.rstrip().endswith("exit 0")
    for printed in (
        "sys.version",  # the Python version
        "uv --version",
        "setup_environment",  # the venv creation and the package install
        "ldd",  # mediapipe's native modules, when the import fails
        "df -h",  # free disk space
        'method="HEAD"',
    ):
        assert printed in source, printed
    assert builder.archive_url() in source
    assert "get_source" not in source.split("# ---- end of the common part ----")[1]


def bash() -> str | None:
    found = shutil.which("bash")
    # Windows' C:\Windows\System32\bash.exe starts WSL, which cannot run this Python.
    return found if found and "system32" not in found.lower() else None


def test_clean_drops_the_kaggle_kernel_variables(tmp_path):
    """The cause of the failed Kaggle runs: the kernel's MPLBACKEND (matplotlib_inline) reached
    the venv, where mediapipe's import of matplotlib.pyplot rejected it."""
    if bash() is None:
        pytest.skip("no bash")
    common = builder.common()
    script = tmp_path / "clean.sh"
    script.write_text(
        common
        + 'clean "$PYTHON_UNDER_TEST" -c "import json, os; print(json.dumps(dict(os.environ)))"\n',
        encoding="utf-8",
        newline="\n",
    )
    env = dict(
        os.environ,
        EV_WORK=str(tmp_path / "work"),
        EV_TMP=str(tmp_path / "tmp"),
        EV_INPUT=str(tmp_path / "input"),
        EV_PART="1/3",
        MPLBACKEND="module://matplotlib_inline.backend_inline",
        PYTHONPATH="/kaggle/lib/kagglegym",
        UV_SYSTEM_PYTHON="true",
        KAGGLE_USER_SECRETS_TOKEN="secret",
        PYTHON_UNDER_TEST=sys.executable,
    )
    done = subprocess.run([bash(), str(script)], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    assert seen["MPLBACKEND"] == "Agg" and seen["PYTHONNOUSERSITE"] == "1"
    assert seen["EV_PART"] == "1/3" and "EV_INPUT" in seen and "PATH" in seen
    for gone in (
        "PYTHONPATH",
        "UV_SYSTEM_PYTHON",
        "KAGGLE_USER_SECRETS_TOKEN",
        "PYTHON_UNDER_TEST",
    ):
        assert gone not in seen, gone


@pytest.mark.parametrize("name", list(builder.notebook_paths()))
def test_every_cell_takes_the_code_from_the_archive_only(name):
    source = cell(name)
    assert source.startswith("%%bash\n")
    assert source.rstrip().splitlines()[-1] in ("finish", "exit 0")
    assert f'SOURCE_FINGERPRINT="{builder.source_fingerprint()}"' in source
    assert "edge-vision-src*.tar.gz" in source and "CODE SOURCE MISSING" in source
    assert "git clone" not in source and "github.com" not in source
    assert source.count("get_source") >= 2  # defined and called


@pytest.mark.parametrize(
    ("name", "limit", "part", "dataset"),
    [
        ("asl_citizen_smoke", 20, "0/1", False),
        ("asl_citizen_part1", 0, "0/3", False),
        ("asl_citizen_part2", 0, "1/3", False),
        ("asl_citizen_part3", 0, "2/3", False),
        ("asl_citizen_merge", 0, "0/1", True),
    ],
)
def test_the_extraction_cells(name, limit, part, dataset):
    source = cell(name)
    assert f"export EV_LIMIT_VIDEOS={limit} " in source
    assert f'export EV_PART="{part}" ' in source
    assert ("make_private_dataset.py" in source) is dataset
    assert 'uv pip install --python "$py" mediapipe==0.10.21 opencv-python-headless' in source
    assert "ENVIRONMENT FAILED" in source and 'clean "$PY" -u' in source
    assert "--script-out" in source and "def build_mapping(" not in source


def test_the_training_cell_checks_its_python_and_has_a_venv_fallback():
    source = cell("train_v3")
    assert builder.COMMON_MARKER not in source
    assert "deps_ok" in source and "train_venv" in source and "ENVIRONMENT FAILED" in source
    assert "scripts/v3_kaggle.py locate" in source
    assert "\n    python scripts/" not in source  # the training steps run through py()


def test_script_out_writes_only_the_extraction_program(tmp_path):
    out = tmp_path / "extract.py"
    assert builder.main(["--script-out", str(out)]) == 0
    program = out.read_text("utf-8")
    labels = json.loads((ROOT / "web/public/model/meta.json").read_text("utf-8"))["labels"]
    assert "def build_mapping(" in program and json.dumps(labels) in program
    compile(program, str(out), "exec")
