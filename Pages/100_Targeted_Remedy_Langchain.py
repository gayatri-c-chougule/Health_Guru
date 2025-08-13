#region Imports
import streamlit as st
from langchain_remedy import find_remedy
#endregion 

#region st1 — UI for ailment, body type, and remedy type selection
st.markdown(
    "<h1 style='text-align: center; color: #6B4D31;'>🍀 Health Guru AI - Targeted & Precise Remedies</h1>",
    unsafe_allow_html=True  # Required to allow HTML/CSS styling in Streamlit text
)

# Main input: ailment description
ailment_description = st.text_area("Describe your ailment and symptoms:")

# Split selection options into two columns for better UI layout
col1, col2 = st.columns(2)

with col1:
    body_type_options = ["General", "Pitta", "Kapha", "Vata"]
    cmb_body_type = "Choose your body type"
    sel_body_type = st.selectbox(cmb_body_type, body_type_options)

with col2:
    # Remedy options list matches internal logic in `find_remedy` & prompt rules
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

# Helpful tips for users to get better search results
with st.expander("💡 Tips for better results"):
    st.markdown("""
    - If you don't receive results for your selected **body type**, try selecting **"General"**.
    - If a specific **remedy type** doesn't return results, try:
        - Choosing **"Overall"**, or  
        - Exploring other **remedy types** such as Dietary, Herbal, or Exercise options.
    """)
#endregion

#region st2 — Handle button click to trigger remedy search
if st.button("Find"):
    with st.spinner("Loading..."):
        # `find_remedy` handles the LangChain + vector search logic internally
        result = find_remedy(ailment_description, sel_remedy_type, sel_body_type)
        st.text_area("Remedy", result, height=400)
#endregion

#region st Body Type Desc — navigation hint
st.markdown(
    """🧬Not sure about your body type? Head to the "Find Body Type" page to compare traits and find the one that best matches you!"""
)
#endregion
