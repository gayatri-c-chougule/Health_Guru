#region Imports — core libs and LangChain/LangGraph
import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langchain_community.vectorstores import FAISS
from langchain.schema import SystemMessage, HumanMessage
#endregion

#region Loading — models, vector store, retriever
load_dotenv()

# Mild variation while staying stable; for metric runs set to 0.0 for determinism.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # default if not set
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))  # default if not set

llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# NOTE: allow_dangerous_deserialization=True is required for FAISS index metadata with LangChain.
# Only load from trusted paths to avoid code execution risks.
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH") 
vector_store = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)

# We use k=12 based on internal testing; good recall without too much irrelevant context.
retriever_remedy = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 12},
)
#endregion

#region Graph Init — state schema
class State(TypedDict):
    """
    Graph state passed between nodes.

    Keys:
        ailment_description: User input describing symptoms/condition.
        body_type: Ayurvedic body type ('Vata', 'Pitta', 'Kapha', 'General').
        remedy_type: Requested remedy type ('Herbal', 'Dietary', 'Yoga', 'Overall').
        context: Joined text from retrieved documents.
        response: Final or intermediate response text.
        is_specific: True when both body_type and remedy_type are not general/overall.
        stored_remedy_type: Original remedy_type (kept for fallback routing).
    """
    ailment_description: str
    body_type: str
    remedy_type: str
    context: str
    response: str  # Final output shown to the user after all graph logic completes
    is_specific: bool  # to check if the body type and remedy type are both specific and not general
    stored_remedy_type: str  # preferred remedy type stored separately; fallback logic may modify remedy_type


class Context(TypedDict, total=False):
    """Optional runtime context for LangGraph (unused here)."""
    pass
#endregion

#region GraphNodes — core node functions
def check_specificity(state: State, runtime: Runtime[Context]) -> dict:
    """
    Mark the query as specific when both body_type and remedy_type are not general/overall.

    Args:
        state (State): Contains `body_type` and `remedy_type`.
        runtime (Runtime[Context]): LangGraph runtime (unused).

    Returns:
        dict: {"is_specific": True} if both are specific; otherwise {}.
    """
    body = (state.get("body_type") or "").lower()
    rtype = (state.get("remedy_type") or "").lower()
    return {"is_specific": True} if body != "general" and rtype != "overall" else {}

def retrieve_context(state: State, runtime: Runtime[Context]) -> dict:
    """
    Retrieve documents for the ailment and set the concatenated text as context.

    Args:
        state (State): Requires `ailment_description`.
        runtime (Runtime[Context]): LangGraph runtime (unused).

    Returns:
        dict: {"context": <joined document text>}.
    """
    docs = retriever_remedy.invoke(state.get("ailment_description", ""))
    # Join with a generator to avoid building an intermediate list.
    context = "\n\n".join(doc.page_content for doc in docs).strip()
    return {"context": context}

def generate_remedy_node(state: State, runtime: Runtime[Context]) -> dict:
    """
    Generate a remedy using the provided context, body_type, and remedy_type.

    Args:
        state (State): Uses `context`, `ailment_description`, `remedy_type`, `body_type`.
        runtime (Runtime[Context]): LangGraph runtime (unused).

    Returns:
        dict: {"response": <LLM output or 'No remedy found.'>}.
    """
    system_prompt = """You are an expert Ayurvedic practitioner.
Use ONLY the information provided in the CONTEXT to answer the user's query.
Do NOT guess or invent information.
If no remedy is found in the context, reply exactly:
"No remedy found."

Remedy Logic:
- If body type is "Vata", "Pitta", or "Kapha", find remedies specific to that body type.
- If body type is "General", find remedies for any body type.
- If remedy type is "Overall", return remedies of any type.
- If remedy type is specified (e.g., Herbal, Dietary, Yoga), return remedies matching that type.
- If no remedy matches both body type and remedy type, reply:
"No remedy found."
"""
    user_prompt = f"""CONTEXT:
{state.get("context", "")}

USER QUERY:
Symptoms or Disease: {state.get("ailment_description", "")}
Requested Remedy Type: {state.get("remedy_type", "")}
Body Type: {state.get("body_type", "")}

Answer:"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages).content
    # Keep exact sentinel match for downstream routing.
    return {"response": response}

def reroute_query_node(state: State, runtime: Runtime[Context]) -> dict:
    """
    Broaden body_type and/or remedy_type in steps when no remedy is found.

    Args:
        state (State): Contains `body_type`, `remedy_type`, `is_specific`, `stored_remedy_type`.
        runtime (Runtime[Context]): LangGraph runtime (unused).

    Returns:
        dict: Updated fields and/or a terminal "response".
    """
    remedy_type = (state.get("remedy_type") or "").lower()
    body_type = (state.get("body_type") or "").lower()
    is_specific = state.get("is_specific")
    stored_remedy_type = state.get("stored_remedy_type")

    # Body type: General and Remedy type: Overall
    # No remedies found for fully general query → stop with terminal "None".
    if body_type == "general" and remedy_type == "overall":
        return {"response": "None"}

    # Body type: General and Remedy type: Specific → relax remedy type first.
    elif body_type == "general" and remedy_type != "overall":
        return {"remedy_type": "overall", "response": "finding"}

    # Body type: Specific and Remedy type: Overall (not originally specific) → relax body type.
    elif body_type != "general" and remedy_type == "overall" and is_specific is False:
        return {"body_type": "general", "response": "finding"}

    # Originally specific → relax body type but restore original remedy_type.
    elif body_type != "general" and remedy_type == "overall" and is_specific is True:
        return {"body_type": "general", "remedy_type": stored_remedy_type, "response": "finding"}

    # Body type: Specific and Remedy type: Specific → relax remedy type.
    elif body_type != "general" and remedy_type != "overall":
        return {"remedy_type": "overall", "response": "finding"}


def final_response_node(state: State, runtime: Runtime[Context]) -> dict:
    """
    Format the final remedy text for the user.

    Args:
        state (State): Contains `body_type`, `remedy_type`, and `response`.
        runtime (Runtime[Context]): LangGraph runtime (unused).

    Returns:
        dict: {"response": <formatted string>}.
    """
    return {
        "response": (
            f"For body type: {state.get('body_type')} and remedy type: {state.get('remedy_type')}, "
            f"Remedy: {state.get('response')}"
        )
    }
#endregion

#region Graph wiring — nodes & edges
graph = StateGraph(state_schema=State)
graph.add_node("check_specificity", check_specificity)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("generate_remedy_node", generate_remedy_node)
graph.add_node("reroute_query_node", reroute_query_node)
graph.add_node("final_response_node", final_response_node)
#endregion

#region Graph flow — entry, conditionals, finish
# First check specificity → retrieve context → attempt remedy generation.
graph.set_entry_point("check_specificity")
graph.add_edge("check_specificity", "retrieve_context")
graph.add_edge("retrieve_context", "generate_remedy_node")

def check_remedy_found(state: State) -> str:
    """
    Decide next step based on LLM output.

    Returns:
        "no_remedy_found" if the response is exactly the sentinel "No remedy found.",
        else "remedy_found".
    """
    # Exact string match is intentional; keep it synchronized with the system prompt.
    return "no_remedy_found" if state["response"] == "No remedy found." else "remedy_found"

graph.add_conditional_edges(
    source="generate_remedy_node",
    path=check_remedy_found,
    path_map={
        "no_remedy_found": "reroute_query_node",
        "remedy_found": "final_response_node",
    },
)

def check_after_rerouting(state: State) -> str:
    """
    After rerouting, either finalize or try generation again.

    Returns:
        "no_remedy_found" if terminal sentinel "None" is set,
        else "finding" to loop back to generation.
    """
    # "None" denotes we've exhausted fallbacks and found nothing.
    return "no_remedy_found" if state["response"] == "None" else "finding"

graph.add_conditional_edges(
    source="reroute_query_node",
    path=check_after_rerouting,
    path_map={
        "no_remedy_found": "final_response_node",
        "finding": "generate_remedy_node",
    },
)

graph.set_finish_point("final_response_node")
compiled = graph.compile()
#endregion

#region Public API
def get_remedy_graph():
    """Return the compiled LangGraph for external use (e.g., Streamlit or tests)."""
    return compiled
#endregion
