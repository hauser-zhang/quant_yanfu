from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def make_run_dirs(run_dir: Path, enable_debug: bool = False) -> Dict[str, Path]:
    """Create and return standard experiment output directories."""
    run_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        "run": run_dir,
        "figures": run_dir / "figures",
        "tables": run_dir / "tables",
        "factor_ic": run_dir / "factor_ic",
        "ablation": run_dir / "ablation",
        "importance": run_dir / "importance",
    }
    if enable_debug:
        dirs["debug"] = run_dir / "debug"
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def save_config(run_dir: Path, config_dict: dict) -> None:
    """Save experiment configuration to config.json."""
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame table to CSV with parent dirs created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_figure(fig, path: Path) -> None:
    """Save matplotlib figure to disk and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
