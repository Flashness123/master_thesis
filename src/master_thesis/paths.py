"""
Storage layout below $STABLEWM_HOME (the only place that knows it).

$STABLEWM_HOME/
├── arc_environments/                   ARC game files (managed by arc_agi)
├── recordings/<game>/<collection>/     raw JSONL episodes + collection.json
├── datasets/<game>/<dataset>.lance     converted Lance datasets (+ <dataset>.json provenance)
├── evaluation/<game>/                  evaluation outputs, file names start with the dataset/model name
└── models/<kind>/<run>/                everything of one trained model; kind = "lewm" or "ppo"

<game> is the base game id without version ("ls20-9607627b" -> "ls20").
Collections and model runs end with a local MMDD-HHMM timestamp.
"""

import os
from datetime import datetime
from pathlib import Path

MODEL_KINDS = ("lewm", "ppo")  # allowed sub-folders of models/


def stablewm_home() -> Path:
  """Return the data root from the STABLEWM_HOME environment variable."""
  try:
    return Path(os.environ["STABLEWM_HOME"])  # e.g. /home/lukas/TU_Dresden/master_thesis/data/stable_worldmodel
  except KeyError as exc:  # variable not exported in this shell
    raise RuntimeError("STABLEWM_HOME is not set") from exc  # clearer than a bare KeyError


def game_folder(game_id: str) -> str:
  """Folder name of a game: the id without its version hash."""
  return game_id.split("-")[0]  # "ls20-9607627b" -> "ls20"; "ls20" stays "ls20"


def timestamp() -> str:
  """Local time as MMDD-HHMM, appended to collection and run names."""
  return datetime.now().strftime("%m%d-%H%M")  # e.g. "0916-1432"


def arc_environments_dir() -> Path:
  """Folder where arc_agi looks for (and downloads) game files."""
  return stablewm_home() / "arc_environments"


def recordings_dir(game_id: str) -> Path:
  """Folder holding all recording collections of one game."""
  return stablewm_home() / "recordings" / game_folder(game_id)  # e.g. recordings/ls20


def dataset_path(game_id: str, name: str) -> Path:
  """Lance table path of a dataset of one game."""
  return stablewm_home() / "datasets" / game_folder(game_id) / f"{name}.lance"  # e.g. datasets/ls20/goose_l1-7.lance


def evaluation_dir(game_id: str) -> Path:
  """Folder for evaluation outputs (tables, plots, GIFs) of one game; file names start with the dataset or model name."""
  return stablewm_home() / "evaluation" / game_folder(game_id)  # e.g. evaluation/ls20


def model_dir(kind: str, name: str) -> Path:
  """Folder of one trained model run."""
  if kind not in MODEL_KINDS:  # guard against typos such as "lewn"
    raise ValueError(f"Unknown model kind {kind!r}; expected one of {MODEL_KINDS}")
  return stablewm_home() / "models" / kind / name  # e.g. models/lewm/ls20_goose-l1-7_10ep_0914-0027
