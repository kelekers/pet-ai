from __future__ import annotations

import argparse

import uvicorn

from bridge.ipc_server import PRESET_BUILDERS, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="PET-AI WebSocket Cognitive Bridge Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--preset",
        default="impossible_cat",
        choices=sorted(PRESET_BUILDERS.keys()),
    )
    parser.add_argument("--hz", type=float, default=6.0, help="Cognitive ticks per second.")
    parser.add_argument(
        "--live-ollama",
        action="store_true",
        help="Use local Ollama (qwen3:1.7b) off-thread for live skill synthesis.",
    )
    parser.add_argument("--model", default="qwen3:1.7b")
    args = parser.parse_args()

    app = create_app(
        preset=args.preset,
        tick_hz=args.hz,
        auto_tick=True,
        use_live_ollama=args.live_ollama,
        ollama_model=args.model,
        async_synthesis=True,
    )
    print(f"PET-AI Bridge listening on ws://{args.host}:{args.port}/ws (preset={args.preset})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()