"""Repository configuration and experiment logging."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_repo_dotenv() -> None:
    """Load optional local settings; explicit shell variables take precedence."""
    load_dotenv(REPO_ROOT / ".env", override=False)


def init_wandb_run(args: argparse.Namespace, *, is_main_process: bool):
    if not is_main_process or args.report_to != "wandb":
        return None

    import wandb

    kwargs = {
        "project": os.getenv("WANDB_PROJECT") or "hint-sd",
        "entity": os.getenv("WANDB_ENTITY") or None,
        "name": args.run_name,
        "config": {**vars(args), "algorithm": "hint-sd"},
    }
    if args.wandb_run_id:
        kwargs.update(id=args.wandb_run_id, resume="must")
    return wandb.init(**kwargs)
