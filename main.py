from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import settings
from utils.logging import get_logger, setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Realtime / STT / TTS evaluation system"
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="Run scenario (eval_type in YAML: realtime | stt | tts)",
    )
    run.add_argument("--scenario", required=True, help="Path to scenario YAML")
    run.add_argument("--variation", default=None)
    run.add_argument("--agent", default=None, help="Realtime agent override")
    run.add_argument(
        "--provider",
        default=None,
        help="STT/TTS provider override (groq|openai|google|sarvam)",
    )
    run.add_argument("--model", default=None, help="Provider model override")
    run.add_argument("--voice", default=None, help="TTS voice override")
    run.add_argument(
        "--stt",
        default=None,
        help="STT provider for realtime agent transcript (or STT eval)",
    )
    run.add_argument(
        "--tts",
        default=None,
        help="TTS provider for realtime user audio (or TTS eval)",
    )

    lst = sub.add_parser("list-providers", help="List registered adapters")
    lst.add_argument(
        "--kind",
        choices=["all", "realtime", "stt", "tts"],
        default="all",
    )

    dash = sub.add_parser("dashboard", help="Watch-only live dashboard")
    dash.add_argument("--host", default=None)
    dash.add_argument("--port", type=int, default=None)

    sub.add_parser("init-db", help="Create / migrate SQLite schema")
    return p


async def cmd_run(args: argparse.Namespace) -> int:
    from engine.runner import result_to_dict, run_eval

    result = await run_eval(
        args.scenario,
        variation=args.variation,
        agent=args.agent,
        provider=args.provider,
        model=args.model,
        voice=args.voice,
        stt=args.stt,
        tts=args.tts,
    )
    payload = result_to_dict(result)
    get_logger("main").info("Done %s", payload)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_list_providers(kind: str) -> int:
    from agents import list_adapters
    from stt import list_stt
    from tts import list_tts

    if kind in ("all", "realtime"):
        print("realtime:", ", ".join(list_adapters()))
    if kind in ("all", "stt"):
        print("stt:", ", ".join(list_stt()))
    if kind in ("all", "tts"):
        print("tts:", ", ".join(list_tts()))
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    import uvicorn

    host = args.host or settings.dashboard_host
    port = args.port or settings.dashboard_port
    uvicorn.run(
        "dashboard.app:app",
        host=host,
        port=port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
    return 0


def cmd_init_db() -> int:
    from db.schema import init_db

    conn = init_db(settings.db_path)
    conn.close()
    print(f"Initialized DB at {settings.db_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return asyncio.run(cmd_run(args))
    if args.command == "list-providers":
        return cmd_list_providers(args.kind)
    if args.command == "dashboard":
        return cmd_dashboard(args)
    if args.command == "init-db":
        return cmd_init_db()
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
