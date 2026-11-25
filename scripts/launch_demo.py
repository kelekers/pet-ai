from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from bridge.ipc_server import PRESET_BUILDERS

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch PET-AI Cognitive Bridge, Godot 4 3D Studio, and Telemetry Inspector."
    )
    parser.add_argument(
        "--preset",
        default="impossible_cat",
        choices=sorted(PRESET_BUILDERS.keys()),
        help="Initial creature genome preset.",
    )
    parser.add_argument("--hz", type=float, default=6.0, help="Simulation ticks per second.")
    parser.add_argument("--host", default="127.0.0.1", help="Bridge server bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bridge server port.")
    parser.add_argument(
        "--live-ollama",
        action="store_true",
        help="Enable live local Ollama LLM synthesis instead of deterministic fallback.",
    )
    parser.add_argument("--model", default="qwen3:1.7b", help="Ollama model name.")
    parser.add_argument(
        "--godot-path",
        default=None,
        help="Explicit path to the Godot 4 executable.",
    )
    parser.add_argument(
        "--no-godot",
        action="store_true",
        help="Run only the Python bridge and browser inspector without launching Godot 4.",
    )
    parser.add_argument(
        "--open-ui",
        action="store_true",
        help="Open http://127.0.0.1:<port>/inspector in the default browser once ready.",
    )
    return parser


def find_godot_executable(explicit_path: str | Path | None = None) -> Path | None:
    """Locate the Godot 4 executable via explicit flag, env var, PATH, or Windows install paths."""
    if explicit_path is not None and str(explicit_path).strip():
        candidate = Path(explicit_path).expanduser()
        return candidate if candidate.is_file() else None

    env_path = os.environ.get("GODOT_PATH", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return candidate

    for cmd in ("godot4", "godot"):
        found = shutil.which(cmd)
        if found:
            return Path(found)

    known_paths = [
        Path(r"C:\Users\acer\Downloads\Godot_v4.7.1-stable_win64.exe\Godot_v4.7.1-stable_win64.exe"),
    ]
    for path in known_paths:
        if path.is_file():
            return path

    downloads = Path.home() / "Downloads"
    if downloads.is_dir():
        for pattern in ("Godot*/*Godot*.exe", "*Godot*.exe"):
            matches = sorted(downloads.glob(pattern), reverse=True)
            for match in matches:
                if match.is_file() and "console" not in match.name.lower():
                    return match

    prog_files = Path(r"C:\Program Files")
    if prog_files.is_dir():
        for match in sorted(prog_files.glob("Godot*/*Godot*.exe"), reverse=True):
            if match.is_file():
                return match

    return None


def wait_for_health(url: str, timeout_sec: float = 25.0) -> bool:
    """Poll the bridge /health endpoint until HTTP 200 or timeout."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.25)
    return False


def _stop_process(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bridge_script = REPO_ROOT / "scripts" / "run_bridge_server.py"
    godot_project = REPO_ROOT / "godot_client"

    bridge_cmd = [
        sys.executable,
        str(bridge_script),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--preset",
        args.preset,
        "--hz",
        str(args.hz),
        "--model",
        args.model,
    ]
    if args.live_ollama:
        bridge_cmd.append("--live-ollama")

    print(f"Starting PET-AI Cognitive Bridge on http://{args.host}:{args.port} ...")
    bridge_proc = subprocess.Popen(bridge_cmd, cwd=str(REPO_ROOT))
    health_url = f"http://{args.host}:{args.port}/health"
    inspector_url = f"http://{args.host}:{args.port}/inspector"

    try:
        if not wait_for_health(health_url, timeout_sec=30.0):
            print(f"Error: Bridge server did not become healthy at {health_url}.")
            return 1

        print(f"Bridge ready! Inspector UI: {inspector_url}")
        if args.open_ui:
            webbrowser.open(inspector_url)

        if args.no_godot:
            print("Running in bridge-only mode (--no-godot). Press Ctrl+C to stop.")
            bridge_proc.wait()
            return 0

        godot_exe = find_godot_executable(args.godot_path)
        if godot_exe is None:
            print(
                "Warning: Godot 4 executable not found. Pass --godot-path or set GODOT_PATH.\n"
                f"Bridge remains active for the browser inspector at {inspector_url}. "
                "Press Ctrl+C to stop."
            )
            bridge_proc.wait()
            return 0

        print(f"Launching Godot 4 Studio ({godot_exe}) ...")
        godot_proc = subprocess.Popen(
            [str(godot_exe), "--path", str(godot_project)],
            cwd=str(REPO_ROOT),
        )
        godot_proc.wait()
        return 0
    except KeyboardInterrupt:
        print("\nShutting down PET-AI demo...")
        return 0
    finally:
        _stop_process(bridge_proc)


if __name__ == "__main__":
    raise SystemExit(main())
