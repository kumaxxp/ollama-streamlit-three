#!/usr/bin/env python3
"""
Install Ollama models based on config/model_config.json

Usage examples:
  - List planned models (defaults: groups=primary,lightweight + defaults):
      python scripts/install_models.py --list
  - Pull planned models (dry-run):
      python scripts/install_models.py --pull --dry-run
  - Pull only primary group, skip models already installed:
      python scripts/install_models.py --pull --groups primary --skip-available --yes
  - Pull explicit models:
      python scripts/install_models.py --pull --names qwen2.5:7b-instruct-q4_K_M gemma3:4b
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Set

CONFIG_PATHS = [
    os.path.join("config", "model_config.json"),
    os.path.join(os.path.dirname(__file__), "..", "config", "model_config.json"),
]


def load_config() -> Dict:
    for p in CONFIG_PATHS:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    print("[WARN] config/model_config.json not found. Using minimal defaults.")
    return {
        "production_models": {},
        "default_models": {},
        "model_selection_rules": {},
    }


def get_installed_models() -> Set[str]:
    """Return set of installed model names via `ollama list`.
    Uses CLI to avoid hard dependency on Python client in all environments.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return set()
        lines = result.stdout.strip().splitlines()
        names: Set[str] = set()
        for line in lines[1:]:  # skip header
            parts = line.split()
            if parts:
                names.add(parts[0])
        return names
    except Exception:
        return set()


def build_plan(config: Dict, groups: List[str], include_defaults: bool, extra_names: List[str]) -> List[str]:
    """Build an ordered unique list of model names to install based on config.
    Priority order:
      1) default_models (agent, director, fallback) if include_defaults
      2) production_models selected groups, sorted by item.priority asc
      3) extra explicit names
    """
    plan: List[str] = []
    seen: Set[str] = set()

    def add(name: str):
        if name and name not in seen:
            seen.add(name)
            plan.append(name)

    if include_defaults:
        dm = config.get("default_models", {})
        for key in ("agent", "director", "fallback"):
            v = dm.get(key)
            if v:
                add(v)

    prod = config.get("production_models", {})
    # resolve groups; if groups == ["all"], include all groups present
    selected_groups = list(prod.keys()) if (not groups or groups == ["all"]) else groups
    # flatten
    items = []
    for g in selected_groups:
        arr = prod.get(g)
        if isinstance(arr, list):
            items.extend(arr)
    # sort by priority if present
    items.sort(key=lambda x: (x.get("priority") is None, x.get("priority", 1_000_000)))
    for it in items:
        add(it.get("name"))

    for n in extra_names:
        add(n)

    return plan


def pull_models(models: List[str], skip_available: bool, yes: bool, dry_run: bool) -> int:
    installed = get_installed_models() if skip_available else set()
    exit_code = 0

    for name in models:
        if skip_available and name in installed:
            print(f"[SKIP] already installed: {name}")
            continue
        cmd = ["ollama", "pull", name]
        if dry_run:
            print(f"[DRY-RUN] {' '.join(cmd)}")
            continue
        if not yes:
            try:
                ans = input(f"Pull {name}? [Y/n] ").strip().lower()
            except EOFError:
                ans = "y"
            if ans and ans not in ("y", "yes"):  # default yes if empty
                print(f"[SKIP] user cancelled: {name}")
                continue
        print(f"[RUN] {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[ERROR] failed: {name} (code={rc})")
            exit_code = rc
    return exit_code


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Install Ollama models from config")
    action = parser.add_mutually_exclusive_group(required=False)
    action.add_argument("--list", action="store_true", help="List planned models and exit")
    action.add_argument("--pull", action="store_true", help="Pull planned models")

    parser.add_argument(
        "--groups",
        default="primary,lightweight",
        help="Comma-separated production model groups to include (or 'all')",
    )
    parser.add_argument(
        "--include-defaults",
        action="store_true",
        help="Include default_models (agent/director/fallback) at the beginning",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=[],
        help="Explicit model names to include additionally",
    )
    parser.add_argument("--skip-available", action="store_true", help="Skip models already installed")
    parser.add_argument("--yes", "-y", action="store_true", help="Assume yes for prompts")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args(argv)

    config = load_config()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()] if args.groups else []

    plan = build_plan(config, groups, args.include_defaults, args.names)

    if args.list or not (args.list or args.pull):
        print("Planned models:")
        for i, n in enumerate(plan, 1):
            print(f"  [{i}] {n}")
        print("\nHint: add --pull to install, and --dry-run to preview commands.")
        return 0

    # pull
    return pull_models(plan, skip_available=args.skip_available, yes=args.yes, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
