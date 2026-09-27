from __future__ import annotations

import argparse
import json
from pathlib import Path

from .hf_download import download_model, repo_gguf_listing
from .inventory import build_inventory
from .profile_resolver import resolved_inventory, resolve_profiles
from .server_manager import list_servers, prepare_launch_command, server_logs, stop_server


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", help="Optional existing llama.cpp/control-center root.")
    parser.add_argument(
        "--model-dir",
        action="append",
        default=[],
        help="Model directory to scan. Repeat to scan more than one directory.",
    )
    parser.add_argument("--max-files", type=int, default=10000, help="Maximum GGUF files to include.")
    parser.add_argument("--no-manifest", action="store_true", help="Skip models.json profile import.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")


def inventory_command(args: argparse.Namespace) -> int:
    payload = build_inventory(
        project_root=args.project_root,
        model_dirs=[Path(path) for path in args.model_dir] or None,
        include_manifest=not args.no_manifest,
        max_files=args.max_files,
    )
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def profiles_command(args: argparse.Namespace) -> int:
    profiles = resolve_profiles(
        project_root=args.project_root,
        model_dirs=[Path(path) for path in args.model_dir] or None,
    )
    payload = {
        "profiles": [profile.to_dict() for profile in profiles],
        "launchable_count": len([profile for profile in profiles if profile.launchable]),
    }
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def resolved_inventory_command(args: argparse.Namespace) -> int:
    payload = resolved_inventory(
        project_root=args.project_root,
        model_dirs=[Path(path) for path in args.model_dir] or None,
    )
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def prepare_command(args: argparse.Namespace) -> int:
    result = prepare_launch_command(
        mode=args.mode,
        project_root=args.project_root,
        model_dirs=[Path(path) for path in args.model_dir] or None,
    )
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0 if result.get("success") else 2


def servers_command(args: argparse.Namespace) -> int:
    payload = {"servers": list_servers()}
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def stop_command(args: argparse.Namespace) -> int:
    result = stop_server(server_id=args.server_id, mode=args.mode)
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0 if result.get("success") else 2


def logs_command(args: argparse.Namespace) -> int:
    result = server_logs(args.server_id, args.lines)
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0 if result.get("success") else 2


def hf_files_command(args: argparse.Namespace) -> int:
    result = repo_gguf_listing(args.repo_id, revision=args.revision)
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0 if result.get("success") else 2


def pull_command(args: argparse.Namespace) -> int:
    result = download_model(
        args.repo_id,
        pattern=args.pattern,
        quant=args.quant,
        include_mmproj=not args.no_mmproj,
        dest_dir=args.dest,
        revision=args.revision,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0 if result.get("success") else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m inferencedeck",
        description="Portable discovery and inventory tools for InferenceDeck.",
    )
    subparsers = parser.add_subparsers(dest="command")

    inventory_parser = subparsers.add_parser("inventory", help="Print runtime/model/profile inventory as JSON.")
    _add_common_args(inventory_parser)
    inventory_parser.set_defaults(func=inventory_command)

    profiles_parser = subparsers.add_parser("profiles", help="Resolve runnable profiles against discovered models.")
    _add_common_args(profiles_parser)
    profiles_parser.set_defaults(func=profiles_command)

    resolved_parser = subparsers.add_parser("resolved-inventory", help="Print inventory plus resolved profile matches.")
    _add_common_args(resolved_parser)
    resolved_parser.set_defaults(func=resolved_inventory_command)

    prepare_parser = subparsers.add_parser("prepare", help="Build the llama-server command for a profile without launching.")
    _add_common_args(prepare_parser)
    prepare_parser.add_argument("mode", help="Profile mode from models.json.")
    prepare_parser.set_defaults(func=prepare_command)

    servers_parser = subparsers.add_parser("servers", help="List tracked servers launched by the portable core.")
    servers_parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    servers_parser.set_defaults(func=servers_command)

    stop_parser = subparsers.add_parser("stop", help="Stop a tracked server by id or profile mode.")
    stop_parser.add_argument("--server-id")
    stop_parser.add_argument("--mode")
    stop_parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    stop_parser.set_defaults(func=stop_command)

    logs_parser = subparsers.add_parser("logs", help="Read logs for a tracked server.")
    logs_parser.add_argument("server_id")
    logs_parser.add_argument("--lines", type=int, default=200)
    logs_parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    logs_parser.set_defaults(func=logs_command)

    hf_files_parser = subparsers.add_parser("hf-files", help="List the GGUF quants and mmproj files in a Hugging Face repo.")
    hf_files_parser.add_argument("repo_id", help="Hugging Face repo, e.g. unsloth/gemma-3-4b-it-GGUF.")
    hf_files_parser.add_argument("--revision", default="main")
    hf_files_parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    hf_files_parser.set_defaults(func=hf_files_command)

    pull_parser = subparsers.add_parser("pull", help="Download one GGUF quant (all shards, plus its mmproj) from Hugging Face.")
    pull_parser.add_argument("repo_id", help="Hugging Face repo, e.g. unsloth/gemma-3-4b-it-GGUF.")
    pull_parser.add_argument("--quant", help="Quant to fetch, e.g. Q4_K_M.")
    pull_parser.add_argument("--pattern", help="Filename glob, e.g. '*UD-Q4_K_XL*'.")
    pull_parser.add_argument("--no-mmproj", action="store_true", help="Skip the vision projector.")
    pull_parser.add_argument("--dest", help="Download into this folder instead of the Hugging Face cache.")
    pull_parser.add_argument("--revision", default="main")
    pull_parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded.")
    pull_parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    pull_parser.set_defaults(func=pull_command)

    _add_common_args(parser)
    parser.set_defaults(func=inventory_command)

    args = parser.parse_args(argv)
    return args.func(args)
