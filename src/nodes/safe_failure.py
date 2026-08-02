from typing import Dict, Any
from src.state import AgentState, FinalOutput, ClassificationType


def safe_failure_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node for safe failure fallback when maximum verification attempts are reached."""
    fallback_output = FinalOutput(
        classification=ClassificationType.SAFE_FAILURE,
        answer="I am unable to provide a verified answer at this time. Your request has been logged for support review.",
        sources=[],
        confidence=0.0,
        requires_human=True,
        reason="Maximum verification retry limit reached without passing schema/guardrail checks.",
        warnings=["Verification failed after maximum retry attempts."]
    )
    return {
        "final_output": fallback_output,
        "classification": ClassificationType.SAFE_FAILURE
    }
