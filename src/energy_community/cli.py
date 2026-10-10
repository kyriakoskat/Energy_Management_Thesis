"""Single entry point for the thesis framework and individual experiment stages."""

import argparse
from .config import load_config
from .experiments import run, DEPENDENCIES


def main():
    parser = argparse.ArgumentParser(
        description="Run the energy-community thesis framework"
    )
    parser.add_argument(
        "--config", default="configs/demo.json", help="JSON configuration file"
    )
    parser.add_argument(
        "--stage",
        choices=["all", *DEPENDENCIES],
        default="all",
        help="Stage and its dependencies (default: all)",
    )
    parser.add_argument(
        "--output",
        help="Override the output directory; paths relative to the current directory",
    )
    args = parser.parse_args()
    try:
        config, base = load_config(args.config)
        directory = run(config, base, args.stage, args.output)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"\nResults: {directory}")
