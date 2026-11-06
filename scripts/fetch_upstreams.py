"""Materialize locked public sources; never install or execute their scripts."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaic_distill.upstreams import ROOT, verify_upstream

lock = json.loads((ROOT / "upstreams.lock.json").read_text())
for name, spec in lock.items():
    destination = ROOT / "external" / name
    if not destination.exists():
        destination.mkdir(parents=True)
        def git(*args):
            subprocess.run(["git", "-C", str(destination), *args], check=True)
        git("init")
        git("remote", "add", "origin", spec["url"])
        git("config", "remote.origin.promisor", "true")
        git("config", "remote.origin.partialclonefilter", "blob:none")
        git("fetch", "--depth", "1", "--filter=blob:none", "origin", spec["commit"])
        git("sparse-checkout", "init", "--no-cone")
        git("sparse-checkout", "set", "--no-cone", *spec["sparse_patterns"])
        git("checkout", "--detach", "FETCH_HEAD")
    verify_upstream(name)
    print(f"{name}: verified {spec['commit']}")
