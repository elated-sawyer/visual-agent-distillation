"""Mandatory outer entrypoint for rendering queues; no implicit retries."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaic_distill.render_guard import supervise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(supervise(command, args.run_dir, root/'runs/render-safety'))
