# Create, or add a version to, the PRIVATE Kaggle dataset asl-citizen-landmarks-private from a
# finished extraction (kaggle/asl_citizen_merge.ipynb runs it from the edge-vision-src code
# archive). Runs with Kaggle's own Python, where kaggle_secrets lives:
#
#     python make_private_dataset.py /kaggle/working/asl-citizen-landmarks
#
# It reads the KAGGLE_USERNAME and KAGGLE_KEY notebook secrets, writes ~/.kaggle/kaggle.json and
# calls "kaggle datasets create -p OUT --dir-mode zip", which makes a private dataset; --public is
# never passed (the ASL Citizen licence forbids redistribution, docs/DATA_LICENSES.md). If the
# dataset exists it adds a version. When the secrets are missing or a call fails it prints the
# exact manual steps instead. It always exits 0, like the notebook cell.

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

SLUG = "asl-citizen-landmarks-private"
FAILURE = re.compile(r"\b(error|failed|forbidden|unauthori[sz]ed|40[0-9])\b", re.IGNORECASE)

Runner = Callable[[list[str], dict[str, str]], tuple[int, str]]


def manual_steps(out: Path, reason: str) -> str:
    return f"""
DATASET NOT CREATED AUTOMATICALLY: {reason}
Make it by hand. It must stay Private: the ASL Citizen licence forbids redistribution.
  1. Wait until this version has finished, open it and go to its Output.
  2. Choose New Dataset (or Datasets > New Dataset > Notebook Output Files) and select the
     folder {out.name}.
  3. Title: {SLUG}. Visibility: Private (the default). Create.
Or skip the dataset: in kaggle/train_v3.ipynb, Add Input > Your Work > Notebooks > this notebook
(this version). The training notebook finds manifest.csv under either kind of input.
To make the dataset automatically next time: Add-ons > Secrets, add KAGGLE_USERNAME and
KAGGLE_KEY (kaggle.com > Settings > API > Create New Token gives kaggle.json), tick both for this
notebook, then commit this merge run again.
""".strip()


def notebook_secrets() -> tuple[str, str]:
    from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

    client = UserSecretsClient()
    return client.get_secret("KAGGLE_USERNAME"), client.get_secret("KAGGLE_KEY")


def write_credentials(user: str, key: str, home: Path) -> Path:
    directory = home / ".kaggle"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "kaggle.json"
    path.write_text(json.dumps({"username": user, "key": key}))
    path.chmod(0o600)
    return path


def kaggle_cli() -> list[str] | None:
    found = shutil.which("kaggle")
    if not found:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=False)
        local = Path.home() / ".local" / "bin" / "kaggle"
        found = shutil.which("kaggle") or (str(local) if local.is_file() else None)
    return [found] if found else None


def run(command: list[str], env: dict[str, str]) -> tuple[int, str]:
    done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    return done.returncode, (done.stdout + "\n" + done.stderr).strip()


def succeeded(code: int, text: str) -> bool:
    return code == 0 and not FAILURE.search(text)


def make_dataset(
    out: Path,
    *,
    secrets: Callable[[], tuple[str, str]] = notebook_secrets,
    runner: Runner = run,
    cli: Callable[[], list[str] | None] = kaggle_cli,
    home: Path | None = None,
) -> bool:
    """True when the private dataset was created or versioned; otherwise prints manual steps."""
    if not (out / "COMPLETE").is_file():
        print(manual_steps(out, f"{out} has no COMPLETE file, so the extraction is not finished"))
        return False
    if (out / "SMOKE").exists():
        print(manual_steps(out, "this is a smoke-test output, which is never a dataset"))
        return False
    try:
        user, key = secrets()
        if not user or not key:
            raise ValueError("a secret is empty")
    except Exception as error:
        reason = f"the KAGGLE_USERNAME / KAGGLE_KEY secrets are not available ({error})"
        print(manual_steps(out, reason))
        return False
    credentials = write_credentials(user, key, home or Path.home())
    print(f"wrote {credentials} (mode 600) for Kaggle user {user}")
    dataset_id = f"{user}/{SLUG}"
    metadata = {"title": SLUG, "id": dataset_id, "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=1))
    command = cli()
    if not command:
        print(manual_steps(out, "the kaggle command-line tool could not be installed"))
        return False
    env = dict(os.environ, KAGGLE_CONFIG_DIR=str(credentials.parent))
    # "datasets create" makes a private dataset unless --public is passed; it is never passed.
    create = [*command, "datasets", "create", "-p", str(out), "--dir-mode", "zip"]
    print("$ kaggle " + " ".join(create[len(command) :]))
    code, text = runner(create, env)
    print(text)
    if not succeeded(code, text) and re.search(r"already (in use|exists)", text, re.IGNORECASE):
        version = [*command, "datasets", "version", "-p", str(out), "--dir-mode", "zip", "-m",
                   "asl_citizen_merge run"]  # fmt: skip
        print(f"{dataset_id} exists; adding a version")
        code, text = runner(version, env)
        print(text)
    if not succeeded(code, text):
        print(manual_steps(out, f"the kaggle command failed (exit {code}; output above)"))
        return False
    _, status = runner([*command, "datasets", "status", dataset_id], env)
    print(f"status of {dataset_id}: {status}")
    print(
        f"private dataset: https://www.kaggle.com/datasets/{dataset_id} - processing the upload "
        "can take a few minutes; check that its page says Private"
    )
    return True


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("/kaggle/working/asl-citizen-landmarks")
    try:
        make_dataset(out)
    except Exception as error:  # the cell always exits 0; say what happened instead
        print(manual_steps(out, f"unexpected {type(error).__name__}: {error}"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
