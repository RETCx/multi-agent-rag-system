from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from src.agents import create_data_retriever, create_report_generator
from src.utils import _extract_text


class AgentState(TypedDict):
    query: str      # user input
    snippets: list[str]  # filled by Data Retriever
    answer: str     # filled by Report Generator


def build_graph():
    """Build and compile the LangGraph StateGraph."""
    
    retriever_instance = create_data_retriever()
    generator_instance = create_report_generator()

    def retriever_node(state: AgentState) -> dict:
        snippets = retriever_instance(state["query"])
        return {"snippets": snippets}
    
    def generator_node(state: AgentState) -> dict:
        snippets = state["snippets"]
        print(f"[Report Generator] Generating answer from {len(snippets)} snippet(s)...")

        if not snippets:
            context_string = "No relevant information found in the knowledge base."
        else:
            context_string = "\n\n".join([f"[Snippet {i+1}]\n{s}" for i, s in enumerate(snippets)])

        result = generator_instance.invoke({
            "query": state["query"],
            "snippets": context_string,  
        })
        answer = _extract_text(result.content)
        return {"answer": answer}

        
    graph = StateGraph(AgentState)
    graph.add_node("data_retriever", retriever_node)
    graph.add_node("report_generator", generator_node)
    graph.add_edge(START, "data_retriever")
    graph.add_edge("data_retriever", "report_generator")
    graph.add_edge("report_generator", END)
    
    return graph.compile()
