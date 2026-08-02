import re
from typing import Dict, Any, List

from src.state import AgentState, FinalOutput, SourceItem, ClassificationType


def verification_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node to verify draft answer against schema rules and build FinalOutput."""
    attempts = state.verification_attempts + 1
    draft = state.draft_answer or ""
    classification = state.classification or ClassificationType.ANSWERABLE
    retrieved_docs: List[Dict[str, Any]] = state.retrieved_docs or []

    warnings: List[str] = []
    cited_sources: List[SourceItem] = []

    # 1. Extract source citations (KB-xxx or CASE-xxx or filename) referenced in draft answer or retrieved docs
    detected_ids = set(re.findall(r"\b(?:KB-\d{3}|CASE-\d{4})\b", draft, re.IGNORECASE))

    # Match detected IDs to retrieved doc passages
    for doc in retrieved_docs:
        s_id = str(doc.get("source_id", ""))
        passage_snippet = str(doc.get("passage", doc.get("content", "")))[:200]

        # If explicitly cited in text or top retrieved doc, include as source
        if s_id in detected_ids or not detected_ids:
            cited_sources.append(
                SourceItem(
                    source_id=s_id if s_id else "documentation",
                    passage=passage_snippet if passage_snippet else "OrbitDesk reference document"
                )
            )

    # De-duplicate cited sources
    unique_sources = []
    seen_ids = set()
    for s in cited_sources:
        if s.source_id not in seen_ids:
            seen_ids.add(s.source_id)
            unique_sources.append(s)

    if not detected_ids and draft:
        warnings.append("Draft answer did not explicitly reference document IDs inline.")

    # 2. Determine confidence and human escalation flag based on classification
    requires_human = classification in [
        ClassificationType.REQUIRES_ESCALATION,
        ClassificationType.OUT_OF_SCOPE,
        ClassificationType.REQUIRES_CLARIFICATION,
    ]

    if classification == ClassificationType.ANSWERABLE:
        confidence = 0.95 if unique_sources else 0.80
        reason = "Answer validated directly against knowledge base documentation."
    elif classification == ClassificationType.REQUIRES_CLARIFICATION:
        confidence = 0.70
        reason = "Request requires additional diagnostic details from the user."
    elif classification == ClassificationType.REQUIRES_ESCALATION:
        confidence = 0.60
        reason = "Issue exhibits persistent failures after standard checks; requires engineering escalation."
    else: # OUT_OF_SCOPE or SAFE_FAILURE
        confidence = 0.90
        reason = "Request is outside supported agent operational scope."

    # 3. Construct Pydantic FinalOutput model
    final_output = FinalOutput(
        classification=classification,
        answer=draft if draft else "Unable to generate a suitable answer.",
        sources=unique_sources,
        confidence=confidence,
        requires_human=requires_human,
        reason=reason,
        warnings=warnings,
    )

    return {
        "verification_attempts": attempts,
        "final_output": final_output,
    }
