import time
from openai import BadRequestError
from graph import build_graph
from queries import QUERIES


def run_query(graph, query: str) -> None:
    """Run a single query through the multi-agent pipeline and print the result."""
    print(f"\n{'=' * 55}")
    print(f"QUERY: {query}")
    print(f"{'=' * 55}")

    try:
        result = graph.invoke({"query": query})
        print(f"\n[Final Answer]\n{result['answer']}")
    except BadRequestError as e:
        # Azure content filter returns HTTP 400 BadRequestError.
        print(f"\n[Final Answer]")
        print(f"Request rejected by Azure content management policy: {e.code}")
        print("  The information is not available in the knowledge base.")

    print(f"{'=' * 55}\n")


def main():
    graph = build_graph()

    for i, q in enumerate(QUERIES):
        run_query(graph, q)
        if i < len(QUERIES) - 1:
            print("  [waiting 65s for rate limit...]\n")
            time.sleep(65)


if __name__ == "__main__":
    main()
