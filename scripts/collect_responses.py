"""Collect one model response at a time for the conversation-context benchmark."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-5.2"
DEFAULT_CULTURES = ("chinese", "afghan", "nepali", "maori", "burmese", "vietnamese")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Collect conversation-context model responses sequentially through a chat API."
    )
    parser.add_argument(
        "--conversations-dir",
        required=True,
        type=Path,
        help="Directory containing <culture>/persona_*/background_*_conversation_* files.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=repo_root / "data/queries/filtered_queries_final.csv",
        help="Filtered query CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="JSONL file to which response records are appended.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CCBENCH_MODEL", DEFAULT_MODEL),
        help=f"Chat model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.together.xyz/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("TOGETHER_API_KEY", os.getenv("OPENAI_API_KEY")),
        help="API key; defaults to TOGETHER_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--cultures",
        nargs="+",
        default=list(DEFAULT_CULTURES),
        choices=DEFAULT_CULTURES,
        help="Cultures to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for shuffling each persona's conversation history.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow duplicate custom_ids in an existing output file.",
    )
    return parser.parse_args()


def conversation_text(path: Path) -> str | None:
    content = path.read_text(encoding="utf-8")
    try:
        content = content.split("_conv>", 1)[1]
        content = content.split("<", 1)[0].strip()
    except IndexError:
        print(f"Skipping malformed conversation file: {path}")
        return None
    return content


def conversation_messages(persona_dir: Path, seed: int | None) -> list[dict[str, str]]:
    files = sorted(
        glob.glob(str(persona_dir / "background_health_conversation_*.json"))
        + glob.glob(str(persona_dir / "background_nonhealth_conversation_*.txt"))
    )
    if seed is not None:
        random.Random(seed).shuffle(files)
    else:
        random.shuffle(files)

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for filename in files:
        content = conversation_text(Path(filename))
        if content is None:
            continue
        if "[PERSONA]" not in content:
            print(f"Skipping conversation without persona turns: {filename}")
            continue
        turns = [turn.strip() for turn in content.split("[PERSONA]")[1:]]
        turns = [part for turn in turns for part in turn.split("[AGENT]")]
        for index, turn in enumerate(turns):
            if not turn:
                continue
            messages.append(
                {"role": "user" if index % 2 == 0 else "assistant", "content": turn}
            )

    messages.append(
        {
            "role": "user",
            "content": (
                "Now answer the following healthcare-related query based on the conversation "
                "history, carefully considering the culturally-relevant health-related norms "
                "and behaviors that the user has revealed."
            ),
        }
    )
    return messages


def existing_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    ids = set()
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if isinstance(record.get("custom_id"), str):
                    ids.add(record["custom_id"])
    return ids


def collect(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise SystemExit("Set TOGETHER_API_KEY, OPENAI_API_KEY, or pass --api-key.")

    import pandas as pd
    from openai import OpenAI

    queries = pd.read_csv(args.queries)
    completed = set() if args.overwrite else existing_ids(args.output)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as output:
        for culture in args.cultures:
            culture_dir = args.conversations_dir / culture
            persona_dirs = sorted(culture_dir.glob("persona_*"))
            if not persona_dirs:
                raise FileNotFoundError(f"No persona directories found in {culture_dir}")

            for persona_dir in persona_dirs:
                persona_id = persona_dir.name.rsplit("_", 1)[-1]
                history = conversation_messages(persona_dir, args.seed)
                for _, row in queries.iterrows():
                    question_id = int(row["question_id"])
                    custom_id = f"{args.model}_convcontext_{culture}_{persona_id}_{question_id}"
                    if custom_id in completed:
                        print(f"Skipping completed record: {custom_id}")
                        continue

                    messages = history + [{"role": "user", "content": str(row["question"])}]
                    print(f"Requesting {custom_id}")
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=messages,
                        temperature=args.temperature,
                    )
                    record: dict[str, Any] = {
                        "custom_id": custom_id,
                        "response": {"body": response.model_dump()},
                    }
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    completed.add(custom_id)


if __name__ == "__main__":
    arguments = parse_args()
    collect(arguments)
