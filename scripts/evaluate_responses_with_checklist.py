"""Evaluate generated healthcare responses through a chat-completions API."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-5.2"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate response records with a chat-completions API."
    )
    parser.add_argument(
        "--responses",
        required=True,
        help="JSONL file containing generated healthcare responses.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSONL file to which evaluation results will be appended.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CCBENCH_MODEL", DEFAULT_MODEL),
        help=f"Chat model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="Optional OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key; defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--queries",
        default=repo_root / "data/queries/filtered_queries_final.csv",
        help="Filtered query CSV.",
    )
    parser.add_argument(
        "--personas-dir",
        default=repo_root / "data/personas",
        help="Directory containing culture/persona JSON files.",
    )
    parser.add_argument(
        "--checklists-dir",
        default=repo_root / "evaluations/checklists",
        help="Directory containing culture checklist JSON files.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--system-prompt",
        dest="system_prompt_file",
        default=repo_root / "system_prompts/checklist_evaluation.txt",
        help="Checklist system prompt text file.",
    )
    return parser.parse_args()


def response_text(record: dict[str, Any]) -> str:
    try:
        content = record["response"]["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Record has no response.body.choices[0].message.content") from error
    return str(content).replace("```json", "").replace("```", "").strip()


def record_context(record: dict[str, Any]) -> tuple[str, str, int]:
    custom_id = record.get("custom_id")
    if not isinstance(custom_id, str):
        raise ValueError("Record has no string custom_id")
    match = re.search(r"_(?P<culture>[a-z]+)_(?P<persona>[^_]+)_(?P<question>\d+)$", custom_id)
    if not match:
        raise ValueError(
            "custom_id must end with _<culture>_<persona_id>_<question_id>: "
            f"{custom_id}"
        )
    return match["culture"], match["persona"], int(match["question"])


def recommendations_for(
    record: dict[str, Any],
    queries: pd.DataFrame,
    personas_dir: Path,
    checklists_dir: Path,
) -> list[dict[str, str]]:
    culture, persona_id, question_id = record_context(record)
    matching_rows = queries[queries["question_id"] == question_id]
    if matching_rows.empty:
        raise ValueError(f"No query found for question_id={question_id}")
    query_row = matching_rows.iloc[0]

    persona_file = personas_dir / culture / f"persona_{persona_id}.json"
    checklist_file = checklists_dir / f"{culture}_checklist.json"
    with persona_file.open(encoding="utf-8") as handle:
        persona_norms = json.load(handle)
    with checklist_file.open(encoding="utf-8") as handle:
        checklist = json.load(handle)

    persona_norms = [
        norm
        for norm in persona_norms
        if norm["adherence"] != "Neutral"
        and norm["norm_id"] in query_row.index
        and bool(query_row[norm["norm_id"]])
    ]
    adherence_by_norm = {norm["norm_id"]: norm["adherence"] for norm in persona_norms}

    recommendations = []
    for item in checklist:
        norm_id = item["norm_id"]
        adherence = adherence_by_norm.get(norm_id)
        if adherence is None:
            continue
        recommendation_key = (
            "follow_recommendation" if adherence == "Follow" else "avoid_recommendation"
        )
        recommendations.append(
            {
                "norm_id": norm_id,
                "norm": item["norm"],
                "recommendation": item[recommendation_key],
            }
        )
    return recommendations


def evaluate_record(
    client: OpenAI,
    record: dict[str, Any],
    queries: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    recommendations = recommendations_for(
        record,
        queries,
        Path(args.personas_dir),
        Path(args.checklists_dir),
    )
    prompt = (
        f"Response: {response_text(record)}\n Recommendations: ```"
        f"{json.dumps(recommendations)}```"
    )
    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=args.temperature,
    )
    content = completion.choices[0].message.content or ""
    return {
        "custom_id": record["custom_id"],
        "recommendations": recommendations,
        "evaluation": content.replace("```json", "").replace("```", "").strip(),
        "model": args.model,
    }


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")

    import pandas as pd
    from openai import OpenAI

    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)
    queries = pd.read_csv(args.queries)

    with open(args.responses, encoding="utf-8") as responses, open(
        args.output, "a", encoding="utf-8"
    ) as output:
        for index, line in enumerate(responses):
            if not line.strip():
                continue
            record = json.loads(line)
            result = evaluate_record(client, record, queries, args)
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(f"Evaluated record {index}: {record['custom_id']}")


if __name__ == "__main__":
    main()
