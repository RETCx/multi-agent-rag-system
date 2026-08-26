import time
from graph import build_graph

def run_query(graph, query: str) -> None:
    """Run a single query through the multi-agent pipeline and print the result."""
    print(f"\n{'=' * 55}")
    print(f"QUERY: {query}")
    print(f"{'=' * 55}")

    result = graph.invoke({"query": query})

    print(f"\n[Final Answer]\n{result['answer']}")
    print(f"{'=' * 55}\n")


def main():
    graph = build_graph()

    queries = [
        "What is the policy on international travel?",
        "What are the remote work options available?",
        "What products does the company offer?",
        "What is the meal allowance for domestic and international travel?",
        "What certifications does SecureID have?",
        "What is the company's policy on cryptocurrency investment?",
    ]

    for i, q in enumerate(queries):
        run_query(graph, q)
        if i < len(queries) - 1:
            print("  [waiting 65s for rate limit...]\n")
            time.sleep(65)  


if __name__ == "__main__":
    main()
