from typing import Dict, Any
from langgraph.graph import StateGraph, END

from src.state import AgentState, ClassificationType
from src.nodes import (
    triage_node,
    retrieval_node,
    generation_node,
    verification_node,
    safe_failure_node,
)


def route_triage(state: AgentState) -> str:
    """Conditional router after Triage node.
    Routes to 'retrieval' if classification is 'answerable', otherwise routes to 'verification'
    (exit node to format non-answerable responses before ending).
    """
    if state.classification == ClassificationType.ANSWERABLE or str(state.classification).lower() == "answerable":
        return "retrieval"
    return "verification"


def is_verification_successful(state: AgentState) -> bool:
    """Checks whether the verification node passed validation."""
    if not state.final_output:
        return False
    if state.classification != ClassificationType.ANSWERABLE and str(state.classification).lower() != "answerable":
        return True
    answer = state.final_output.answer.strip()
    if not answer or answer == "Unable to generate a suitable answer.":
        return False
    return True


def route_verification(state: AgentState) -> str:
    """Conditional router after Verification node.
    Loops backward to 'generation' if verification fails AND attempts < 2.
    Routes to 'safe_failure' if verification fails AND attempts >= 2.
    Routes to 'end' if verification succeeds.
    """
    if is_verification_successful(state):
        return "end"
    if state.verification_attempts < 2:
        return "generation"
    return "safe_failure"


def build_graph():
    """Builds and compiles the StateGraph workflow for support agent orchestration."""
    workflow = StateGraph(AgentState)

    # 1. Add Triage, Retrieval, Generation, Verification, and Safe Failure nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("safe_failure", safe_failure_node)

    # Set entry point to Triage
    workflow.set_entry_point("triage")

    # 2. Conditional edge after Triage (retrieval if 'answerable', otherwise exit node 'verification')
    workflow.add_conditional_edges(
        "triage",
        route_triage,
        {
            "retrieval": "retrieval",
            "verification": "verification",
        },
    )

    # Sequential edges for answerable flow
    workflow.add_edge("retrieval", "generation")
    workflow.add_edge("generation", "verification")

    # 3. Conditional edge after Verification (retry generation if < 2 attempts, safe_failure if >= 2 attempts)
    workflow.add_conditional_edges(
        "verification",
        route_verification,
        {
            "generation": "generation",
            "safe_failure": "safe_failure",
            "end": END,
        },
    )

    # Safe failure routes to END
    workflow.add_edge("safe_failure", END)

    return workflow.compile()


# Compiled graph instance
graph = build_graph()
