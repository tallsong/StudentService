import argparse
import json

from student_service.agent import search_xu_university_knowledge


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question against xu-university.com using the local RAG index."
    )
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of retrieved snippets to return",
    )

    args = parser.parse_args()
    question = " ".join(args.question).strip()

    result = search_xu_university_knowledge(question)

    if result.get("status") == "success":
        sources = result.get("sources")
        if isinstance(sources, list):
            result["sources"] = sources[: max(1, args.top_k)]

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
