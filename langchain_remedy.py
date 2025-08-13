#region Imports
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
import os
#endregion 

#region llm
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Loaded via .env so keys aren't hard-coded
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# NOTE: Local FAISS indices created by LangChain may require `allow_dangerous_deserialization=True`.
# This can execute pickled metadata during load—**only** load from trusted paths.
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH") 
vector_store = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

def format_docs(retrieved_docs):
    """
    Format a list of retrieved document objects into a single context string.

    Args:
        retrieved_docs (list): A list of document objects.

    Returns:
        str: A concatenated string of all document contents separated by two newlines,
        or "No relevant reference found." if the list is empty.
    """
    # If retriever returns nothing, be explicit so the LLM knows to fall back.
    if not retrieved_docs:
        return "No relevant reference found."
    # Join page contents with a blank line to avoid merging sentences across docs.
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs).strip()
    return context_text

# We use k=12 as a balance between recall (getting enough varied matches)
# and precision (not flooding the prompt). Tuned empirically for this corpus size.
retriever_remedy = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 12}
)

# For the test cases for the metrics, we set it to 0.0 for consistent results.
# Temperature 0.2 in production allows mild variation without drifting off-spec.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # default if not set
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))  # default if not set
llm_remedy = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
# Prompt strictly enforces "No remedy found." sentinel for routing logic.
prompt_remedy = PromptTemplate(
    input_variables=["context", "ailment_description", "remedy_type", "body_type"],
    template="""
You are an expert Ayurvedic practitioner.

Use ONLY the information provided in the CONTEXT below. Do NOT guess or fabricate information.
If no matching remedy is found, respond with:
**"No remedy found."**

### Remedy Selection Logic:
- If body type is "Vata", "Pitta", or "Kapha", find remedies specific to that type.
- If body type is "General", find remedies suitable for any type.
- If remedy type is "Overall", return remedies of any category.
- If remedy type is specific (e.g., Herbal, Dietary, Yoga), return remedies matching that type.
- If no remedy matches both body type and remedy type, respond with:
**"No remedy found."**

### CONTEXT:
{context}

### USER QUERY:
- Symptoms or Disease: {ailment_description}
- Requested Remedy Type: {remedy_type}
- Body Type: {body_type}

### Answer:
"""
)

# Map inputs → prompt fields; keeps retrieval + formatting separate from LLM call.
chain_remedy = RunnableMap({
    "context": lambda inputs: format_docs(retriever_remedy.invoke( inputs["ailment_description"])),
    "ailment_description": lambda inputs: inputs["ailment_description"],
    "remedy_type": lambda inputs: inputs["remedy_type"],
    "body_type": lambda inputs: inputs["body_type"],
}) | prompt_remedy | llm_remedy | StrOutputParser()

#endregion 

#region function
def find_remedy(ailment_description: str, remedy_type: str, body_type:str):
    """
    Retrieve a remedy based on an ailment description, remedy type, and body type.

    Args:
        ailment_description (str): Description of the ailment or symptoms provided by the user.
        remedy_type (str): Type of remedy requested (e.g., herbal, diet, lifestyle).
        body_type (str): Ayurvedic body type (e.g., Pitta, Kapha, Vata, General).

    Returns:
        str: Suggested remedy text from the chain, or a prompt asking the user to provide an ailment description.
    """
    # Guard against empty inputs so we don't waste tokens or route ambiguous queries.
    if ailment_description.strip():
       formatted_input = {
        "ailment_description": ailment_description,
        "remedy_type": remedy_type,
        "body_type": body_type
    }
       remedy = chain_remedy.invoke(formatted_input)
       return remedy
    else:
        return "Please enter an ailment to get a remedy."
   
#endregion
