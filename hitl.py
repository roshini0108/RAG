"""Human-in-the-loop helpers for the customer support assistant."""

from __future__ import annotations

from typing import Dict, List


def request_human_support(
    query: str,
    draft_answer: str | None,
    reason: str,
    sources: List[str] | None = None,
) -> Dict[str, object]:
    """Simulate escalation to a human reviewer through the command line."""

    print("\n--- Escalation Required ---")
    print(f"Reason: {reason}")
    print(f"Customer Query: {query}")
    if sources:
        print(f"Sources: {sources}")

    if draft_answer:
        print("\nSuggested Draft Answer:")
        print(draft_answer)

    human_answer = input("\nEnter the human-approved response: ").strip()

    if not human_answer:
        human_answer = (
            "A support specialist will review your request and get back to you shortly."
        )

    return {
        "answer": human_answer,
        "handled_by": "human",
        "confidence": 1.0,
        "sources": sources or [],
    }
