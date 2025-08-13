#region Imports
import streamlit as st
from langgraph_remedy import get_remedy_graph
#endregion 

#region st1 — UI inputs for ailment, body type, remedy type
st.markdown(
    "<h1 style='text-align: center; color: #6B4D31;'>🍀 Health Guru AI - Adaptive Remedies</h1>",
    unsafe_allow_html=True  # Needed for HTML/CSS styling in header
)

ailment_description = st.text_area("Describe your ailment and symptoms:")

col1, col2 = st.columns(2)

with col1:
    body_type_options = ["General", "Pitta", "Kapha", "Vata"]
    cmb_body_type = "Choose your body type"
    sel_body_type = st.selectbox(cmb_body_type, body_type_options)

with col2:
    # Remedy type list matches internal graph logic expectations for string matching
    remedy_options = [
        "Overall",
        "Herbal/Ayurvedic medications",
        "Dietary/Nutritional Changes",
        "Yoga Postures/Exercise",
        "Cleansing Procedures",
        "Breathing Exercises"
    ]
    cmb_remedy_type = "Choose your remedy type"
    sel_remedy_type = st.selectbox(cmb_remedy_type, remedy_options)

# Explain fallback routing to the user so they understand why results may broaden.
with st.expander("ℹ️ How fallback search works"):
    st.info("""
    🔁 **Fallback Strategy**

    If no remedy is found for your selected **body type** and **remedy type**, the assistant will try broader combinations:

    1. ✅ Your exact selection  
    2. 🔄 Same body type, any remedy  
    3. 🔄 General body type, your remedy  
    4. 🔄 General body & remedy  
    5. ❌ No remedy found

    This helps ensure you still get the best possible suggestion.
    """)
#endregion

#region Langgraph 
graph = get_remedy_graph()
#endregion

#region st2 — Handle button click to run LangGraph
if st.button("Find"):
    with st.spinner("Loading..."):
        if ailment_description.strip():
            # Initial state matches the LangGraph State schema; 
            # stored_remedy_type preserves the original user choice for fallback routing.
            input_state = {
                "ailment_description": ailment_description,
                "body_type": sel_body_type,
                "remedy_type": sel_remedy_type,
                "context": "",
                "response": "",
                "is_specific": False,
                "stored_remedy_type": sel_remedy_type
            }
            output = graph.invoke(input_state)
            response = output["response"]
            st.text_area("Remedy", response, height=400)
        else:
            st.warning("Please enter an ailment to get a remedy.")
#endregion

#region st Body Type Desc — navigation hint
st.markdown(
    """🧬Not sure about your body type? Head to the "Find Body Type" page to compare traits and find the one that best matches you!"""
)
#endregion
