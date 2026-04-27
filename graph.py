"""LangGraph workflow for the customer support RAG assistant."""

from __future__ import annotations

from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from config import AppConfig, get_config
from hitl import request_human_support
from retrieval import retrieve_documents


class GraphState(TypedDict, total=False):
    """Represent all values carried between LangGraph nodes."""

    query: str
    processed_query: str
    intent: str
    retrieved_docs: List[Document]
    confidence_scores: List[float]
    average_confidence: float
    best_score: float
    sources: List[str]
    route: str
    escalation_reason: str
    draft_answer: str
    answer: str
    handled_by: str
    error: str


def get_llm(config: AppConfig):
    """Create the Ollama chat model used to answer support queries."""

    return ChatOllama(model=config.ollama_model, temperature=0)


def build_prompt() -> ChatPromptTemplate:
    """Return the prompt used to generate grounded support answers."""

    return ChatPromptTemplate.from_template(
        """
You are a helpful customer support assistant.
Use ONLY the retrieved context below to answer the customer question.
Do not use outside knowledge.
Do not guess, invent policies, or fill in missing details.
If the answer is not explicitly supported by the context, reply with exactly:
I don't have enough information.
Keep the answer concise, friendly, and practical.

Retrieved context:
{context}

Customer question:
{question}
"""
    )


def process_query(state: GraphState) -> GraphState:
    """Clean the input query and stop early when the user sends an empty question."""

    raw_query = state["query"].strip()

    if not raw_query:
        return {
            "processed_query": "",
            "route": "error",
            "error": "Please enter a non-empty question.",
        }

    return {
        "processed_query": raw_query,
    }


def classify_intent(state: GraphState) -> GraphState:
    """Classify the query so later nodes can make clearer routing decisions."""

    if state.get("route") == "error":
        return {"intent": "out_of_scope"}

    query = state["processed_query"].lower()
    complex_keywords = ["complaint", "refund", "chargeback", "cancel", "legal", "escalate"]

    if any(keyword in query for keyword in complex_keywords):
        return {"intent": "complex"}

    if len(query.split()) <= 12:
        return {"intent": "simple"}

    return {"intent": "complex"}


def retrieve_docs(state: GraphState) -> GraphState:
    """Fetch the most relevant chunks for the user query."""

    retrieval_result = retrieve_documents(state["processed_query"])
    return {
        "retrieved_docs": retrieval_result["documents"],
        "confidence_scores": retrieval_result["scores"],
        "average_confidence": retrieval_result["average_confidence"],
        "best_score": retrieval_result["best_score"],
        "sources": retrieval_result["sources"],
    }


def decision_node(state: GraphState) -> GraphState:
    """Decide whether to answer automatically or escalate to a human."""

    if state.get("route") == "error":
        print("[DEBUG] Route: ERROR")
        return {"route": "error"}

    if not state.get("retrieved_docs"):
        print("[DEBUG] Route: HITL")
        return {
            "route": "hitl",
            "escalation_reason": "No relevant documents were found.",
            "intent": "out_of_scope",
        }

    # Use the best distance score so one strong match can still answer normally.
    if state.get("best_score", 1.0) > get_config().confidence_threshold:
        print("[DEBUG] Route: HITL")
        return {
            "route": "hitl",
            "escalation_reason": "Best retrieval distance is too high.",
        }

    if state.get("intent") in {"complex", "out_of_scope"}:
        print("[DEBUG] Route: HITL")
        return {
            "route": "hitl",
            "escalation_reason": f"The query was classified as {state.get('intent')}.",
        }

    print("[DEBUG] Route: LLM")
    return {
        "route": "generate_answer",
        "escalation_reason": "",
    }


def generate_answer(state: GraphState) -> GraphState:
    """Generate the final grounded answer from retrieved context."""

    config = get_config()
    llm = get_llm(config)
    prompt = build_prompt()

    context = "\n\n".join(doc.page_content for doc in state.get("retrieved_docs", []))

    try:
        chain = prompt | llm
        response = chain.invoke(
            {
                "context": context,
                "question": state["processed_query"],
            }
        )
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        return {
            "route": "hitl",
            "draft_answer": "",
            "escalation_reason": f"LLM generation failed: {exc}",
        }

    return {
        "draft_answer": answer,
        "answer": answer,
        "handled_by": "llm",
    }


def hitl_node(state: GraphState) -> GraphState:
    """Escalate the case to a simulated human reviewer."""

    human_result = request_human_support(
        query=state["processed_query"],
        draft_answer=state.get("draft_answer"),
        reason=state.get("escalation_reason", "Human review requested."),
        sources=state.get("sources"),
    )
    return {
        "answer": human_result["answer"],
        "handled_by": human_result["handled_by"],
        "sources": human_result["sources"],
        "average_confidence": human_result["confidence"],
    }


def error_node(state: GraphState) -> GraphState:
    """Return a structured error result when the input query is empty."""

    return {
        "answer": state.get("error", "Please enter a non-empty question."),
        "handled_by": "system",
        "sources": [],
        "average_confidence": 0.0,
    }


def route_after_query(state: GraphState) -> Literal["classify_intent", "error"]:
    """Send empty queries to the error node before retrieval begins."""

    return "error" if state.get("route") == "error" else "classify_intent"


def route_decision(state: GraphState) -> Literal["generate_answer", "hitl", "error"]:
    """Return the next node name for conditional routing."""

    route = state.get("route", "hitl")
    if route == "generate_answer":
        return "generate_answer"
    if route == "error":
        return "error"
    return "hitl"


def route_after_generation(state: GraphState) -> Literal["hitl", "__end__"]:
    """Send LLM failures to HITL and successful answers to the end of the graph."""

    return "hitl" if state.get("route") == "hitl" else "__end__"


def build_graph():
    """Build and compile the LangGraph workflow."""

    builder = StateGraph(GraphState)
    builder.add_node("process_query", process_query)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("decision_node", decision_node)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("hitl", hitl_node)
    builder.add_node("error", error_node)

    builder.add_edge(START, "process_query")
    builder.add_conditional_edges("process_query", route_after_query)
    builder.add_edge("classify_intent", "retrieve_docs")
    builder.add_edge("retrieve_docs", "decision_node")
    builder.add_conditional_edges("decision_node", route_decision)
    builder.add_conditional_edges("generate_answer", route_after_generation)
    builder.add_edge("hitl", END)
    builder.add_edge("error", END)

    return builder.compile()
