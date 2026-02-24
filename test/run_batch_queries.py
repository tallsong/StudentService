#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from student_service.agent import search_xu_university_knowledge


def load_questions(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON questions file must be a list of strings.")
        questions = [str(item).strip() for item in data if str(item).strip()]
        return questions

    questions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        questions.append(text)
    return questions


def make_answer(result: dict[str, Any]) -> str:
    status = str(result.get("status", "")).strip().lower()
    if status != "success":
        return str(result.get("message", "No answer available."))

    sources = result.get("sources", [])
    if not isinstance(sources, list) or not sources:
        return "No relevant source found."

    top = sources[0] if isinstance(sources[0], dict) else {}
    content = str(top.get("content", "")).strip()
    if not content:
        return "Retrieved sources, but no content snippet was returned."

    clean = " ".join(content.split())
    max_len = 280
    if len(clean) > max_len:
        return clean[:max_len].rstrip() + "..."
    return clean


def run_queries(questions: list[str], top_k: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    now = datetime.now().isoformat(timespec="seconds")

    for index, question in enumerate(questions, start=1):
        raw = search_xu_university_knowledge(question)
        sources = raw.get("sources", [])
        if isinstance(sources, list):
            raw["sources"] = sources[: max(1, top_k)]

        trimmed_sources = raw.get("sources", [])
        top_source = ""
        if isinstance(trimmed_sources, list) and trimmed_sources:
            first = trimmed_sources[0]
            if isinstance(first, dict):
                top_source = str(first.get("source_url", ""))

        row: dict[str, Any] = {
            "index": index,
            "timestamp": now,
            "question": question,
            "status": raw.get("status"),
            "answer": make_answer(raw),
            "top_source": top_source,
            "source_count": len(trimmed_sources) if isinstance(trimmed_sources, list) else 0,
            "result": raw,
        }
        results.append(row)

    return results


def save_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "timestamp",
                "question",
                "status",
                "answer",
                "top_source",
                "source_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "index": row.get("index"),
                    "timestamp": row.get("timestamp"),
                    "question": row.get("question"),
                    "status": row.get("status"),
                    "answer": row.get("answer"),
                    "top_source": row.get("top_source"),
                    "source_count": row.get("source_count"),
                }
            )


def detect_format(output_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    ext = output_path.suffix.lower()
    if ext == ".json":
        return "json"
    if ext == ".csv":
        return "csv"
    return "jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-test custom questions against StudentService RAG and save answers."
    )
    parser.add_argument(
        "--questions-file",
        default="test/fixtures/questions.txt",
        help="Path to questions file (.txt one per line or .json list).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output file path. Default: test/results/results_<timestamp>.jsonl",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl", "csv"],
        default=None,
        help="Output format. If omitted, inferred from output extension.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Max number of sources to keep per question.",
    )

    args = parser.parse_args()

    questions_file = Path(args.questions_file)
    questions = load_questions(questions_file)
    if not questions:
        raise ValueError(f"No valid questions found in {questions_file}")

    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"test/results/results_{stamp}.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = run_queries(questions, top_k=max(1, args.top_k))
    fmt = detect_format(output_path, args.format)

    if fmt == "json":
        save_json(output_path, rows)
    elif fmt == "csv":
        save_csv(output_path, rows)
    else:
        save_jsonl(output_path, rows)

    print(f"Questions: {len(questions)}")
    print(f"Saved: {output_path}")
    print(f"Format: {fmt}")


if __name__ == "__main__":
    main()
