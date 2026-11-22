# coordinates the whole experiment, from the installation to SUMMARY.md
#
# usage: python3 scripts/pipeline.py [--config configs/config.yaml] [--run-name name]
#
# 1. creates the virtual environment .venv (if it does not exist)
#    and installs requirements.txt, the installation is only
#    repeated when requirements.txt changes and, after it, the
#    script restarts itself with the python of the .venv
# 2. runs the stages of STAGES in order, resuming the ones that
#    already finished, each stage is a run_<stage> function of
#    one of the scripts and runs in its own process
#    the digest of a stage covers its script, the scripts it
#    imports, the configuration, the prompts and the digests of
#    the stages it depends on, the stage is redone when this
#    digest changes, and it goes to the stage (--hash), which
#    uses it to resume only the partial state of the same
#    version (optuna searches, transformer trainings, judge
#    answers and cost records)
#
# besides the orchestrator, this file keeps what every stage
# uses: reading the configuration, the context of a stage
# (folders, tables, alerts), the seeds, the cost measurement
# and the style of the charts
# until step 1 finishes, only the standard library is used

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path

# these libraries only exist after the .venv is
# installed, and the orchestrator works without them
try:
    import matplotlib
    import numpy as np
    import pandas as pd
    import yaml

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    np = None
    pd = None
    yaml = None
    plt = None

# project paths
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
PROMPTS = ROOT / "configs" / "prompts.yaml"
INSTALL_STAMP = VENV / ".pipeline_install.json"
DEFAULT_CONFIG = "configs/config.yaml"
DEFAULT_RUN = "main"

# output folders, relative to the project root
# each run has a name and a subfolder
# with that name in each of them
RESULTS = Path("results")
RUNS = RESULTS / "runs"
OUTPUT_FOLDERS = {
    "tables": RESULTS / "metrics",
    "SUMMARY.md": RESULTS / "metrics",
    "figures": RESULTS / "figures",
    "prediction_tables": RESULTS / "predictions",
    "models": Path("models"),
}

# (stage, script that contains it, stages it depends on)
STAGES = [
    ("hardware", "preparation", []),
    ("data", "preparation", []),
    ("cleaning", "preparation", ["data"]),
    ("lexicon", "preparation", ["cleaning"]),
    ("split", "preparation", ["lexicon"]),
    ("eda", "preparation", ["split"]),
    ("classical", "classical", ["split"]),
    ("transformer", "transformer", ["split"]),
    ("judges_dev", "judges", ["split"]),
    # the test is only opened after the models, thresholds
    # and judge prompts are frozen
    ("test", "evaluation", ["classical", "transformer", "judges_dev"]),
    ("judges_test", "judges", ["judges_dev", "test"]),
    ("comparison", "evaluation", ["test", "judges_test"]),
    ("interpretation", "evaluation", ["test", "judges_test"]),
    ("cost", "evaluation", ["test"]),
    ("report", "evaluation", []),
]
STAGES_ALWAYS_REDONE = {"comparison", "cost", "report"}

# imports of local scripts (from common import ..., import judges),
# including the ones inside functions
# every stage script imports pipeline.py and common.py,
# so a change in them redoes every stage
LOCAL_IMPORT = re.compile(r"^\s*(?:from|import)\s+(\w+)", re.MULTILINE)

# the four phases of the experiment: (identifier, name, stages)
# together, they cover STAGES in the same order
PHASES = [
    ("preprocessing", "Preprocessing", ["hardware", "data", "cleaning", "lexicon"]),
    ("preparation", "Data preparation", ["split", "eda"]),
    ("experiments", "Experiments", ["classical", "transformer", "judges_dev"]),
    (
        "evaluation",
        "Evaluation",
        ["test", "judges_test", "comparison", "interpretation", "cost", "report"],
    ),
]

# number, identifier and name of the phase of a stage
def phase_of(name):
    for number, (key, title, stages) in enumerate(PHASES, start=1):
        if name in stages:
            return number, key, title
    raise ValueError(f"stage {name} is not in PHASES")

# checks if PHASES lists every stage of STAGES, in the same order
def check_phases():
    grouped = []
    for _, _, stages in PHASES:
        for name in stages:
            grouped.append(name)

    ordered = []
    for name, _, _ in STAGES:
        ordered.append(name)

    if grouped != ordered:
        raise SystemExit("PHASES must list every stage of STAGES, in the same order.")

# path of the script that contains the stage
def stage_script(name):
    for stage, script, _ in STAGES:
        if stage == name:
            return SCRIPTS / f"{script}.py"
    raise ValueError(f"unknown stage: {name}")

# the script and the local scripts it imports,
# directly or indirectly
def script_closure(script):
    found = set()
    pending = [script]
    while pending:
        path = pending.pop()
        if path in found or not path.exists():
            continue

        found.add(path)
        text = path.read_text(encoding="utf8")
        for module in LOCAL_IMPORT.findall(text):
            pending.append(SCRIPTS / f"{module}.py")

    return sorted(found)

# sha-256 digest of a stage
# it changes when the configuration, the prompts, the code of
# the stage (with the scripts it imports) or the digest of
# some stage it depends on change
# since the digest of the dependencies goes into the
# calculation, a change in the data of a stage spreads to
# every stage that comes after it
def stage_hash(name, config_text, upstream):
    parts = [config_text]
    if PROMPTS.exists():
        parts.append(PROMPTS.read_text(encoding="utf8"))

    for path in script_closure(stage_script(name)):
        parts.append(path.read_text(encoding="utf8"))

    for digest in upstream:
        parts.append(digest)

    text = "\n".join(parts)
    return hashlib.sha256(text.encode("utf8")).hexdigest()

# saves the status of the stages
# I wrote to a temporary file first so that the status
# does not get corrupted if the program stops in the middle
def save_status(path, status):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf8")
    tmp.replace(path)

# reads the status of the stages, or an empty
# dictionary if the run has not started yet
def load_status(path):
    if not path.exists():
        return {}
    else:
        return json.loads(path.read_text(encoding="utf8"))

# runs a stage in its own process, with
# the output going to the log of the stage
def launch_stage(name, digest, run_dir, config_path):
    command = [
        sys.executable,
        str(stage_script(name)),
        name,
        "--run-dir", str(run_dir),
        "--config", str(config_path),
        "--hash", digest,
    ]
    with open(run_dir / "logs" / f"{name}.log", "a", encoding="utf8") as log:
        return subprocess.run(command, stdout=log, stderr=log).returncode

# command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    return parser.parse_args()

# python of the virtual environment
def venv_python():
    return VENV / "bin" / "python"

# checks if the script is already running inside the .venv
def running_in_venv():
    return Path(sys.prefix).resolve() == VENV.resolve()

# digest of requirements.txt, used to know if it changed
def requirements_signature():
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()

# reads the record of the last installation
# if the file does not exist or is broken, it
# returns an empty dictionary and the installation is redone
def read_install_stamp():
    try:
        return json.loads(INSTALL_STAMP.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return {}

# installs packages with the pip of the .venv
def pip_install(*arguments):
    command = [str(venv_python()), "-m", "pip", "install", "--disable-pip-version-check"]
    command += list(arguments)
    print("[environment] " + " ".join(command), flush=True)
    return subprocess.run(command).returncode

# creates the virtual environment
def create_venv():
    print(f"[environment] creating the virtual environment in {VENV}", flush=True)
    result = subprocess.run([sys.executable, "-m", "venv", str(VENV)])

    if result.returncode != 0 or not venv_python().exists():
        sys.exit(
            "Could not create the .venv. On Ubuntu/Debian, install the module with "
            "'sudo apt install python3-venv' and run it again."
        )

# makes sure the .venv has the dependencies and
# restarts the script inside it
def ensure_environment():
    if not venv_python().exists():
        create_venv()

    # only installs again when requirements.txt changed
    signature = requirements_signature()
    if read_install_stamp().get("signature") == signature:
        print("[environment] dependencies already installed in .venv", flush=True)
    else:
        print("[environment] installing dependencies (only the first time or after changes)", flush=True)
        if pip_install("--requirement", str(REQUIREMENTS)) != 0:
            sys.exit("Installing requirements.txt failed. See the messages above.")

        stamp = {"signature": signature, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
        INSTALL_STAMP.write_text(json.dumps(stamp, indent=2), encoding="utf8")

    # replaces the current process with the same script running in the .venv
    if not running_in_venv():
        python = str(venv_python())
        print(f"[environment] restarting with {python}", flush=True)
        arguments = [python, str(Path(__file__).resolve())] + sys.argv[1:]
        os.execv(python, arguments)

# warns when ollama is not installed
def warn_if_no_ollama():
    if shutil.which("ollama") is None:
        print(
            "WARNING: Ollama not found. The LLM judge stages will fail. "
            "Install it with: curl -fsSL https://ollama.com/install.sh | sh",
            flush=True,
        )

# shows the name of the phase when the stage starts a new phase
def announce_phase(name, current):
    number, key, title = phase_of(name)
    if key != current:
        print(f"\n== Phase {number}/{len(PHASES)}: {title} ==", flush=True)
    return key

# shows how many stages of each phase finished
# and how long they took
def print_phase_summary(status):
    print("\nPhases:")
    for number, (_, title, stages) in enumerate(PHASES, start=1):
        done = 0
        seconds = 0
        for name in stages:
            item = status.get(name, {})
            if item.get("state") == "ok":
                done += 1
            seconds += item.get("seconds") or 0

        print(f"  {number}. {title}: {done}/{len(stages)} stages, {seconds / 60:.1f} min")

# the stage must run again when it is always redone
# or when some stage it depends on was redone
def must_run_again(name, needs, redone):
    if name in STAGES_ALWAYS_REDONE:
        return True

    for need in needs:
        if need in redone:
            return True
    return False

# the stage already finished successfully in this same version
def is_already_done(previous, digest):
    return previous.get("state") == "ok" and previous.get("hash") == digest

# stages this one depends on that have not finished yet
def unfinished_needs(status, needs):
    blocked = []
    for need in needs:
        if status.get(need, {}).get("state") != "ok":
            blocked.append(need)
    return blocked

# runs a stage and records the result in the status
def execute_stage(name, digest, run_dir, config_path, status, status_path):
    print(f"[start] {name}")
    started = time.time()
    begun = time.strftime("%Y-%m-%d %H:%M:%S")

    # marks the stage as running before starting
    status[name] = {"state": "running", "hash": digest, "start": begun, "pid": os.getpid()}
    save_status(status_path, status)

    code = launch_stage(name, digest, run_dir, config_path)
    if code == 0:
        state = "ok"
    else:
        state = "failed"

    status[name] = {
        "state": state,
        "hash": digest,
        "seconds": round(time.time() - started, 1),
        "log": f"logs/{name}.log",
        "start": begun,
        "end": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_status(status_path, status)
    print(f"[{state}] {name}, {status[name]['seconds']} s")
    return state == "ok"

# creates the folder of the run and keeps a copy of the configuration used
def prepare_run_folder(run_name, config_text):
    run_dir = RUNS / run_name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.yaml").write_text(config_text, encoding="utf8")
    print(f"Run: {run_dir}")
    print(f"Metrics: {RESULTS / 'metrics' / run_name}")
    return run_dir

def main():
    os.chdir(ROOT)
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_arguments()
    check_phases()
    ensure_environment()

    config_path = Path(args.config)
    config_text = config_path.read_text(encoding="utf8")
    warn_if_no_ollama()
    run_dir = prepare_run_folder(args.run_name, config_text)

    status_path = run_dir / "status.json"
    status = load_status(status_path)
    redone = set()
    digests = {}
    current_phase = None

    for name, _, needs in STAGES:
        current_phase = announce_phase(name, current_phase)

        # digest of the stage, with the digests of the stages it depends on
        upstream = []
        for need in needs:
            upstream.append(digests[need])
        digest = stage_hash(name, config_text, upstream)
        digests[name] = digest

        # skips the stage that already finished in this same version
        previous = status.get(name, {})
        forced = must_run_again(name, needs, redone)
        if not forced and is_already_done(previous, digest):
            print(f"[resumed] {name} already finished")
            continue

        # does not run the stage if some dependency failed
        blocked = unfinished_needs(status, needs)
        if blocked:
            status[name] = {"state": "blocked", "blocked_by": blocked}
            save_status(status_path, status)
            print(f"[blocked] {name}, depends on {blocked}")
            continue

        if execute_stage(name, digest, run_dir, config_path, status, status_path):
            redone.add(name)

    print_phase_summary(status)

    # stages that did not finish successfully
    pending = []
    for name, item in status.items():
        if item.get("state") != "ok":
            pending.append(name)

    print(f"Done. Summary in {RESULTS / 'metrics' / run_dir.name / 'SUMMARY.md'}")
    if pending:
        print(f"Stages without success: {pending}")
        sys.exit(1)
    else:
        sys.exit(0)

# reads config.yaml
def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf8"))

# reads the prompts of the judges
def load_prompts(path="configs/prompts.yaml"):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf8"))

# sets the seeds for reproducibility
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# expected error of a stage, like a guard that
# stops the experiment from going on
class StageFailed(Exception):
    pass

# where each kind of output of a run
# in results/runs/<name> goes
def output_folders(run_dir):
    run_dir = Path(run_dir)
    folders = {}
    for part, folder in OUTPUT_FOLDERS.items():
        folders[part] = ROOT / folder / run_dir.name
    return folders

# context of a stage: name, run folder,
# configuration, output folders and stage digest
class Context:

    def __init__(self, name, run_dir, config, folders, digest=""):
        self.name = name
        self.run_dir = run_dir
        self.config = config
        self.folders = folders

        # digest of this version of the stage (see stage_hash),
        # which names the state that can be resumed
        self.digest = digest
        self.short_hash = digest[:12]
        self.seed = config["seed"]

    # path of a file of the run, creating the folder if needed
    # tables, figures, models and predictions go to the results
    # and models folders, SUMMARY.md goes to the metrics folder
    # and the rest stays in the run folder
    def path(self, *parts):
        base = self.folders.get(parts[0])
        if base is None:
            path = self.run_dir.joinpath(*parts)
        elif parts[0] in ("tables", "figures", "models", "prediction_tables"):
            path = base.joinpath(*parts[1:])
        else:
            path = base.joinpath(*parts)

        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # shows an alert and keeps it in alerts.csv
    def alert(self, message):
        print(f"ALERT: {message}", flush=True)
        file = self.path("alerts.csv")
        is_new = not file.exists()
        with open(file, "a", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(["stage", "time", "message"])
            writer.writerow([self.name, time.strftime("%Y-%m-%d %H:%M:%S"), message])

    # deletes the old alerts of this stage, so
    # that only the ones of the current run stay
    def clear_alerts(self):
        file = self.path("alerts.csv")
        if not file.exists():
            return

        alerts = pd.read_csv(file)
        alerts[alerts["stage"] != self.name].to_csv(file, index=False)

    # saves a table as csv in the metrics folder
    def save_table(self, table, name):
        table.to_csv(self.path("tables", f"{name}.csv"), index=False)
        print(f"Table saved: {name}.csv ({len(table)} rows)", flush=True)

    # reads a table from the metrics folder
    def read_table(self, name):
        return pd.read_csv(self.path("tables", f"{name}.csv"))

    # checks if the table was already generated
    def has_table(self, name):
        return self.path("tables", f"{name}.csv").exists()

    # saves a dictionary as json
    def save_json(self, data, *parts):
        text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        self.path(*parts).write_text(text, encoding="utf8")

    # reads a json file
    def read_json(self, *parts):
        return json.loads(self.path(*parts).read_text(encoding="utf8"))

    # reads the intermediate data of a stage
    def read_data(self, name):
        return pd.read_parquet(self.path("interim", f"{name}.parquet"))

    # saves the intermediate data of a stage
    def save_data(self, table, name):
        table.to_parquet(self.path("interim", f"{name}.parquet"), index=False)

# creates the context of a stage
def make_context(name, run_dir, config, digest=""):
    run_dir = Path(run_dir)
    return Context(name, run_dir, config, output_folders(run_dir), digest)

# runs a stage of a script
# usage: python scripts/<script>.py <stage> --run-dir ... --config ... --hash ...
def run_stage_cli(stages):
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(stages))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--hash", required=True)
    args = parser.parse_args()

    os.chdir(ROOT)
    config = load_config(args.config)
    set_seeds(config["seed"])
    context = make_context(args.stage, args.run_dir, config, args.hash)
    context.clear_alerts()

    # a guard exits with code 2 and stays in the alerts,
    # and any other error exits with code 1
    try:
        stages[args.stage](context)
    except StageFailed as error:
        print(f"GUARD: {error}", flush=True)
        context.alert(f"GUARD: {error}")
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

# cost: time and vram of each unit of work
# each "with measure(...)" block writes a row in cost/cost_runs.csv
# with the wall clock time and the mean video memory (vram) while
# the block ran, read through nvml every cost.sampling_period_s
# seconds
# the mean_vram_mb column only adds the memory of the processes of
# the method (the stage process, its descendants and, for the
# judges, the ollama processes), and the mean_device_vram_mb column
# has the memory used on the whole card, which includes other
# programs, like the graphical interface and the browser
# without an nvidia card, both stay empty
# the stage_hash column separates the records of the current
# version of each stage from the records of old versions
COST_COLUMNS = [
    "unit",
    "stage",
    "stage_hash",
    "method",
    "scenario",
    "phase",
    "n_documents",
    "start",
    "seconds",
    "mean_vram_mb",
    "mean_device_vram_mb",
    "state",
]

# reads the vram of the first nvidia card, on the whole
# card and in the processes of the method
class VramSampler:

    def __init__(self, ollama=False):
        # I imported them here because pynvml only works with
        # an nvidia card, and whoever creates the sampler handles the failure
        import psutil
        import pynvml

        pynvml.nvmlInit()
        self.nvml = pynvml
        self.psutil = psutil
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.ollama = ollama
        self.own = os.getpid()

        # keeps whether each process belongs to the method, to
        # avoid checking the same process several times
        self.known = {}

    # vram used on the whole card, in mb
    def read_mb(self):
        return self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1e6

    # the process belongs to the method when it is the stage process,
    # a descendant of it or, for the judges, an ollama process
    def check_belongs(self, pid):
        try:
            process = self.psutil.Process(pid)
            if self.ollama and "ollama" in process.name().lower():
                return True
            if pid == self.own:
                return True
            for parent in process.parents():
                if parent.pid == self.own:
                    return True
            return False
        except self.psutil.Error:
            return False

    # same check, using what was already looked up
    def belongs(self, pid):
        if pid not in self.known:
            self.known[pid] = self.check_belongs(pid)
        return self.known[pid]

    # vram of the processes of the method, or None if the card
    # does not report the vram of one of them
    def read_method_mb(self):
        processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        values = []
        for item in processes:
            if self.belongs(item.pid):
                values.append(item.usedGpuMemory)

        # I preferred to record nothing over recording an incomplete sum
        for value in values:
            if value is None:
                return None

        return sum(values) / 1e6

# creates a vram sampler, or returns None if the card
# cannot be read (no nvidia, no driver, etc.)
def make_vram_sampler(ollama=False):
    try:
        return VramSampler(ollama)
    except Exception:
        return None

# reads the vram every period until the measured block finishes
def sample_loop(sampler, readings, stop, period):
    while not stop.is_set():
        readings.append((sampler.read_method_mb(), sampler.read_mb()))
        stop.wait(period)

# mean of one of the columns of the vram readings
# if some reading is missing, the mean stays empty
def mean_reading(readings, position):
    values = []
    for reading in readings:
        values.append(reading[position])

    if not values:
        return None
    for value in values:
        if value is None:
            return None

    return round(float(np.mean(values)), 1)

# appends a row to cost_runs.csv
def append_cost_row(run_dir, row):
    file = Path(run_dir) / "cost" / "cost_runs.csv"
    file.parent.mkdir(parents=True, exist_ok=True)

    # records of a previous version of the pipeline, with
    # other columns, are kept in another file
    if file.exists():
        with open(file, encoding="utf8") as handle:
            header = handle.readline().strip().split(",")
        if header != COST_COLUMNS:
            file.replace(file.with_name(f"cost_runs_old_format_{int(time.time())}.csv"))

    is_new = not file.exists()
    with open(file, "a", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COST_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

# measures the time and the mean vram of the block
# the phase is training, prediction or judgment
# with ollama=True, the vram of the method includes the
# vram of the ollama processes, which serve the judges
@contextmanager
def measure(context, method="", scenario="", phase="", n_documents="", ollama=False):
    sampler = make_vram_sampler(ollama)
    readings = []
    stop = threading.Event()
    thread = None

    # the vram reading runs in a separate thread
    # while the measured block executes
    if sampler is not None:
        thread = threading.Thread(
            target=sample_loop,
            args=(sampler, readings, stop, context.config["cost"]["sampling_period_s"]),
            daemon=True,
        )
        thread.start()

    state = "ok"
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    wall_start = time.perf_counter()

    # the record is written even when the block fails,
    # with the state "error"
    try:
        yield
    except Exception:
        state = "error"
        raise
    finally:
        seconds = time.perf_counter() - wall_start
        stop.set()
        if thread is not None:
            thread.join()

        row = {
            "unit": uuid.uuid4().hex[:12],
            "stage": context.name,
            "stage_hash": context.digest,
            "method": method,
            "scenario": scenario,
            "phase": phase,
            "n_documents": n_documents,
            "start": started_at,
            "seconds": round(seconds, 4),
            "mean_vram_mb": mean_reading(readings, 0),
            "mean_device_vram_mb": mean_reading(readings, 1),
            "state": state,
        }
        append_cost_row(context.run_dir, row)

# processor, memory, system and gpu (through pytorch, if available)
def describe_hardware():
    import psutil

    info = {
        "processor": platform.processor() or platform.machine(),
        "cores": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "system": platform.platform(),
        "python": platform.python_version(),
    }

    # pytorch may not be installed, and in that
    # case the gpu is marked as unavailable
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info["gpu"] = properties.name
            info["gpu_memory_gb"] = round(properties.total_memory / 1e9, 1)
            info["torch_cuda"] = torch.version.cuda
    except ImportError:
        info["cuda_available"] = False

    return info

# checks if there is a gpu with cuda
def gpu_available():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False

# colors and style of the charts
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = "Blues"
INK = "#2b2b29"
MUTED = "#6b6a63"
GRID = "#e4e3dd"
SURFACE = "#ffffff"

# the style is only applied when matplotlib is already installed
if plt is not None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 1,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.bottom": False,
            "ytick.left": False,
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )

# saves the figure as png and closes it
def save_figure(figure, context, name):
    figure.savefig(context.path("figures", f"{name}.png"))
    plt.close(figure)
    print(f"Figure saved: {name}.png", flush=True)

# histogram chart
def histogram(context, name, values, title, xlabel, ylabel="Posts"):
    figure, axis = plt.subplots(figsize=(7, 3.8))
    axis.hist(values, bins=40, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    axis.grid(axis="x", visible=False)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    save_figure(figure, context, name)

# horizontal bar chart, with optional error bars
# and reference line
def horizontal_bars(context, name, labels, values, title, xlabel, errors=None, reference=None,
                    reference_label=None):
    # the height grows with the number of bars
    height = max(2.5, 0.32 * len(labels) + 1.2)
    figure, axis = plt.subplots(figsize=(8, height))
    positions = np.arange(len(labels))

    axis.barh(
        positions,
        values,
        height=0.6,
        color=SERIES[0],
        xerr=errors,
        ecolor=MUTED,
        error_kw={"linewidth": 1}
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.grid(axis="y", visible=False)
    axis.set(xlabel=xlabel)
    axis.set_title(title, pad=20)

    if reference is not None:
        axis.axvline(reference, color=INK, linewidth=1)
        axis.annotate(
            reference_label,
            xy=(reference, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, 3),
            textcoords="offset points",
            color=INK
        )

    save_figure(figure, context, name)

# line chart, one line per series (label, x, y)
def line_chart(context, name, series, title, xlabel, ylabel, reference=None,
               reference_label=None, markers=True):
    figure, axis = plt.subplots(figsize=(7, 4))
    if markers:
        style = {"marker": "o", "markersize": 5}
    else:
        style = {}

    for index, (label, x_values, y_values) in enumerate(series):
        color = SERIES[index % len(SERIES)]
        axis.plot(x_values, y_values, color=color, label=label, **style)

    axis.grid(axis="x", visible=False)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)

    if reference is not None:
        axis.axhline(reference, color=MUTED, linewidth=1)
        axis.annotate(
            reference_label,
            xy=(1, reference),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 3),
            textcoords="offset points",
            color=MUTED,
            ha="right"
        )

    # the legend only shows up with more than one series
    if len(series) > 1:
        axis.legend(loc="upper left", bbox_to_anchor=(1, 1))

    save_figure(figure, context, name)

# heatmap of a square matrix, with the value in each cell
def heatmap(context, name, matrix, labels, title):
    size = max(4, 0.8 * len(labels) + 1.5)
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap=SEQUENTIAL, vmin=-1, vmax=1)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.grid(False)
    axis.set_title(title)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isnan(value):
                continue

            # light text on the dark cells
            if value > 0.45:
                color = SURFACE
            else:
                color = INK
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", color=color)

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    save_figure(figure, context, name)

# 2x2 confusion matrix with the counts and the percentage
# of each row (true class)
# counts[true][classified], with the order [sexist, not sexist]
# in both dimensions
def confusion_matrix_figure(path, counts, title):
    labels = ["Sexist", "Not sexist"]
    counts = np.asarray(counts, dtype=float)

    # percentage of each row, with zero in the empty rows
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    figure, axis = plt.subplots(figsize=(4.6, 4))
    axis.imshow(shares, cmap=SEQUENTIAL, vmin=0, vmax=1)
    axis.set_xticks([0, 1], labels)
    axis.set_yticks([0, 1], labels)
    axis.set_xlabel("Method classification")
    axis.set_ylabel("Human label")
    axis.grid(False)
    axis.set_title(title, fontsize=10)

    for row in range(2):
        for column in range(2):
            # light text on the dark cells
            if shares[row, column] > 0.5:
                color = SURFACE
            else:
                color = INK
            text = f"{int(counts[row, column])}\n{shares[row, column]:.1%}"
            axis.text(column, row, text, ha="center", va="center", color=color)

    figure.savefig(path)
    plt.close(figure)

if __name__ == "__main__":
    main()
