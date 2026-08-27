import argparse
import time
from openai import BadRequestError
from src.graph import build_graph
from src.queries import QUERIES


def run_query(graph, query: str) -> None:
    """Run a single query through the multi-agent pipeline and print the result."""
    print(f"\n{'=' * 55}")
    print(f"QUERY: {query}")
    print(f"{'=' * 55}")

    try:
        result = graph.invoke({"query": query})
        print(f"\n[Final Answer]\n{result['answer']}")
    except BadRequestError as e:
        # Some API gateways (e.g. Azure OpenAI) reject queries that trigger
        # content filters, returning HTTP 400 BadRequestError.
        # This is an optional safety layer — the retrieval threshold and
        # prompt constraint handle injection even without it.
        print(f"\n[Final Answer]")
        print(f"Request rejected by API content policy: {e.code}")
        print("This query was blocked by the API gateway's safety filter.")

    print(f"{'=' * 55}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent RAG System — ask a question about the knowledge base.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-q", "--query",
        type=str,
        help="Run a single custom query (e.g. -q \"What is the meal allowance?\")",
    )
    group.add_argument(
        "-n", "--number",
        type=int,
        metavar="N",
        help=f"Run one of the predefined queries by number (1–{len(QUERIES)})",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all predefined queries (default if no flag given)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=65,
        help="Seconds to wait between queries when running --all (default: 65)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    graph = build_graph()

    # Decide which queries to run
    if args.query:
        queries = [args.query]
    elif args.number:
        if not (1 <= args.number <= len(QUERIES)):
            print(f"Error: --number must be between 1 and {len(QUERIES)}")
            return
        queries = [QUERIES[args.number - 1]]
    else:
        # --all or nothing → run all
        queries = QUERIES

    # Run
    for i, q in enumerate(queries):
        run_query(graph, q)
        if len(queries) > 1 and i < len(queries) - 1:
            print(f"  [waiting {args.delay}s for rate limit...]\n")
            time.sleep(args.delay)


if __name__ == "__main__":
    main()