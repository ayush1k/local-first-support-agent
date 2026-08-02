from typing import Dict, Any, List

from src.state import AgentState
from src.models import get_llm


GENERATION_SYSTEM_PROMPT = """You are a technical support agent for OrbitDesk workspace software.
Your task is to provide a concise, helpful, and accurate response based ONLY on the provided reference documents and resolved cases.

Rules:
1. Ground your answer strictly in the provided reference materials.
2. Cite relevant document IDs (e.g., KB-003, KB-005) or case IDs (e.g., CASE-1041) directly in your response text.
3. If an action cannot be performed (e.g. Viewers creating API tokens), explicitly explain the limitation and required role.
4. Do not invent or assume features or policies not stated in the reference materials.
"""


def generation_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node to draft an answer grounded in retrieved documents."""
    question = state.question
    docs: List[Dict[str, Any]] = state.retrieved_docs or []

    # Format retrieved passages for prompt context
    context_items = []
    for doc in docs:
        source_id = doc.get("source_id", doc.get("id", "DOC"))
        title = doc.get("title", "")
        passage = doc.get("passage", doc.get("content", ""))
        context_items.append(f"Source ID: {source_id}\nTitle: {title}\nContent:\n{passage}")

    context_str = "\n\n---\n\n".join(context_items) if context_items else "No reference documents found."

    prompt = f"Reference Materials:\n{context_str}\n\nUser Question:\n{question}\n\nAnswer:"

    llm = get_llm()
    draft = llm.generate(
        prompt=prompt,
        system_prompt=GENERATION_SYSTEM_PROMPT,
        max_new_tokens=256,
        temperature=0.1,
    )

    return {"draft_answer": draft}
