from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import pandas as pd
import os

#region st1
st.markdown("<h1 style='text-align: center; color: #6B4D31;'>🍀 Ayurvedic Remedies Assistant</h1>", unsafe_allow_html=True)
ailment_description = st.text_area("Describe your ailment and symptoms:")

col1, col2 = st.columns(2)

with col1:
    body_type_options = ["General","Pitta", "Kapha", "Vata"]
    cmb_body_type = "Choose your body type"
    sel_body_type = st.selectbox(cmb_body_type, body_type_options)

with col2:
    remedy_options = ["Overall" ,"Herbal/Ayurvedic medications", "Dietary/Nutritional Changes", "Yoga Postures/Exercise", "Cleansing Procedures", "Breathing Exercises"]
    cmb_remedy_type = "Choose your remedy type"
    sel_remedy_type = st.selectbox(cmb_remedy_type, remedy_options)

with st.expander("💡 Tips for better results"):
    st.markdown("""
    - If you don't receive results for your selected **body type**, try selecting **"General"**.
    - If a specific **remedy type** doesn't return results, try:
        - Choosing **"Overall"**, or  
        - Exploring other **remedy types** such as Dietary, Herbal, or Exercise options.
    """)

#endregion

#region llm
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#r"D:\112_Health_Guru\100_HG_Data\vector_db"
#ec2 path - /home/ec2-user/vector_db
vector_store = FAISS.load_local(
    r"/home/ec2-user/vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

def format_docs(retrieved_docs):
  if not retrieved_docs:
    return "No relevant reference found."
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs).strip()
  return context_text

retriever_remedy = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 12}
)

llm_remedy = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.2)
#res = retriever_remedy.invoke("internal bleeding")

# for doc in res:
#     print("Content:", doc.page_content)
#     print("Metadata:", doc.metadata)
#     print("---")

prompt_remedy = PromptTemplate(
    input_variables=["context", "ailment_description", "remedy_type", "body_type"],
    template="""
You are an expert Ayurvedic practitioner.

Use ONLY the information provided in the CONTEXT below to answer the user's question. Do NOT make up or guess anything. If no remedy is found based on the context, say so clearly.

Follow the instructions exactly:

---

**Remedy Logic:**

1. If the body type is not "General":
   - First, look for remedies that match BOTH the given body type and the remedy type.
   - If none are found:
     - State that no remedies were found for the specific body type.
     - Then look for remedies of the specified remedy type that apply to **all body types**.
     - If still none, look for remedies for any body type, regardless of type.

2. If the body type is "General":
   - Return remedies suitable for any or all body types.

3. If the remedy type is "Overall":
   - Return any relevant remedy that matches the body type logic above.

4. If the remedy type is specific (Herbal medications, Dietary/Nutritional Changes, Yoga Postures, Cleansing Procedures, or Breathing Exercises):
   - Prioritize remedies of that type.
   - If none are found, say so, then provide general remedies of any type if available.

5. If nothing is found, say:
   > "No remedies found based on the provided context."

---

CONTEXT:
{context}

---

USER QUERY:
Symptoms or Disease: {ailment_description}
Requested Remedy Type: {remedy_type}
Body Type: {body_type}

---

Answer:
"""
)

# formatted_input = {
#     "ailment_description": "My nails break easily.",
#     "remedy_type": "Exercise",
#     "body_type": "General"
# }

chain_remedy = RunnableMap({
    "context": lambda inputs: format_docs(retriever_remedy.invoke( inputs["ailment_description"])),
    "ailment_description": lambda inputs: inputs["ailment_description"],
    "remedy_type": lambda inputs: inputs["remedy_type"],
    "body_type": lambda inputs: inputs["body_type"],
}) | prompt_remedy | llm_remedy | StrOutputParser()

# res = chain_remedy.invoke(formatted_input)

#endregion 

#region st2
if st.button("Find"):
   with st.spinner("Loading..."):
    if ailment_description.strip():
       formatted_input = {
        "ailment_description": ailment_description,
        "remedy_type": sel_remedy_type,
        "body_type": sel_body_type
    }
       remedy = chain_remedy.invoke(formatted_input)
       st.text_area("Remedy", remedy, height=400)
    else:
        st.warning("Please enter an ailment to get a remedy.")
    
    #     formatted_input = {
    #         "user_input": ailment_description,
    #         "remedy_type": choice2,
    #         "body_type": choice1
    #     }
    #     remedy = chain.invoke(formatted_input)
    #     st.text_area("Remedy", remedy, height=200)
    # else:
    #     st.warning("Please enter an ailment to get a remedy.")
#endregion

#region st Body Type Desc
st.subheader("🔍 Identify Your Body Type")
st.markdown("""
"Not sure about your body type? Use the table below to compare traits and find the one that best matches you!""")

data = {
   "Trait": [ "Hair" , "Skin" , "Body frame" , "Weight tendency" , "Muscles and bones" , "Eyes" , "Voice" , "Teeth", "Digestion", "Appetite and thirst","Sleep","Body temperature","Mental traits", "Movement style","Sexual drive","Preferred climate",  "Emotional imbalances", "Memory",  "Energy pattern","Food preferences"],
    "Vata": [  "Dry, thin, curly or kinky", "Dry, rough, prone to cracking", "Thin, small, light","Underweight or low weight", "Light muscles, small bones", "Small, recessed, dull or lusterless","Dry, hoarse","Irregular, broken, or protruding",  "Irregular, weak, tendency toward constipation", "Variable appetite and thirst",   "Light, broken, prone to insomnia",  "Cold hands/feet, poor circulation", "Quick, creative, anxious, forgetful", "Fast, restless, erratic",  "Irregular, may experience premature ejaculation", "Warm and moist", "Anxiety, fear, loneliness", "Quick to learn, quick to forget",  "Bursty energy, tires quickly",  "Astringent foods (though needs sweet/sour/salty)"],
    "Pitta":  ["Silky, reddish or early graying, prone to thinning",  "Oily, warm, may have freckles or moles", "Medium, moderate build",  "Stable weight",  "Moderate muscles",  "Bright, sharp, often green/gray/copper-brown",  "Sharp, commanding",  "Sharp, yellowish, bleeding gums",    "Strong digestion and metabolism",  "Strong appetite and thirst",   "Moderate and sound",   "Warm body, sweaty hands and feet",  "Intelligent, focused, can be irritable",  "Sharp and efficient",   "Intense, passionate",   "Cool and dry",   "Anger, impatience, jealousy",  "Sharp and analytical",   "High, sustained energy",    "Spicy, acidic foods (often aggravates them)"],
    "Kapha": ["Thick, soft, wavy, dark, and plentiful", "Soft, smooth, oily, thick, lustrous",  "Large, broad, heavy",   "Easily gains weight, hard to lose",  "Strong muscles, large bones",  "Large, dark, attractive, with long lashes",  "Deep, melodious",  "Large, strong, white",  "Slow digestion",  "Steady appetite, but slow digestion",  "Deep and prolonged",  "Moderate, stable temperature", "Calm, loyal, stable, but can be lethargic when unbalanced",  "Slow, deliberate",  "Strong and sustained without energy loss", "Warm and dry",  "Attachment, greed, envy, lethargy",  "Slow to learn, excellent long-term memory",  "Slow but consistent energy", "Sweet, salty, oily foods (which often aggravate Kapha)"]
}

df = pd.DataFrame(data)
st.table(df.set_index("Trait"))

st.markdown("""
> ⚠️ **Disclaimer**  
> This app is a non-commercial, personal project developed for experimental and educational purposes only, and **not intended for public use**.  
> It uses **sample text** inspired by Ayurvedic principles and is **not created by a certified or trained Ayurvedic practitioner**.  
> The remedies and outputs generated by this tool should **not be considered medical advice**.  
> Always consult a qualified healthcare professional before making any health-related decisions.  
> Use of this content is at your own risk and discretion.
""")
#endregion