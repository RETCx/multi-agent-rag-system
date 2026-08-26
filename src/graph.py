from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from agents import create_data_retriever, create_report_generator
from utils import _extract_text


class AgentState(TypedDict):
    query: str      # user input
    snippets: str   # filled by Data Retriever
    answer: str     # filled by Report Generator


def retriever_node(state: AgentState) -> dict:
    """
    Data Retriever node.

    LLM calls retrieve_from_knowledge_base tool to fetch relevant snippets from the knowledge base.
    Returns raw text snippets (no scores or metadata).
    """
    retriever = create_data_retriever()
    snippets = retriever(state["query"])
    return {"snippets": snippets}


def generator_node(state: AgentState) -> dict:
    """
    Report Generator node.

    Synthesizes retrieved snippets into a final answer.
    Receives only raw text no scores or metadata.
    """
    snippet_count = state["snippets"].count("[Snippet")
    print(f"[Report Generator] Generating answer from {snippet_count} snippet(s)...")

    generator = create_report_generator()
    result = generator.invoke({
        "query": state["query"],
        "snippets": state["snippets"],
    })

    answer = _extract_text(result.content)
    return {"answer": answer}


def build_graph():
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)
    graph.add_node("data_retriever", retriever_node)
    graph.add_node("report_generator", generator_node)
    graph.add_edge(START, "data_retriever")
    graph.add_edge("data_retriever", "report_generator")
    graph.add_edge("report_generator", END)
    return graph.compile()
