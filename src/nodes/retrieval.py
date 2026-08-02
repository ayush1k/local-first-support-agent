from typing import Dict, Any
from src.state import AgentState
from src.store import retrieve_context


def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node to retrieve context documents for the question."""
    docs = retrieve_context(state.question, top_k=5)
    return {"retrieved_docs": docs}
