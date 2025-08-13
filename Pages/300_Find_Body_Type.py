#region Imports
import streamlit as st
import pandas as pd
#endregion

#region Title and Introduction
st.title("🧬 Identify Your Ayurvedic Body Type")

st.markdown("""
Use the guide below to help identify your **Ayurvedic body type** — *Vata*, *Pitta*, or *Kapha* — based on common physical, mental, and physiological traits.

This can help the assistant offer more personalized and balanced remedies.
""")
#endregion

#region Tips
with st.expander("💡 How to Use"):
    st.info("""
    Compare yourself with each column and see which **dosha** best matches your dominant traits.

    Most people are a mix of two types, but identifying your **primary** one helps with remedy suggestions.
    """)
#endregion

#region Body Type Table
# This table is static reference data — not dynamically generated — so it’s safe to render directly.
# Traits are ordered to match typical Ayurvedic assessment categories.
data = {
   "Trait": [
       "Hair", "Skin", "Body frame", "Weight tendency", "Muscles and bones", "Eyes", "Voice", "Teeth",
       "Digestion", "Appetite and thirst", "Sleep", "Body temperature", "Mental traits", "Movement style",
       "Sexual drive", "Preferred climate", "Emotional imbalances", "Memory", "Energy pattern", "Food preferences"
   ],
    "Vata": [
        "Dry, thin, curly or kinky", "Dry, rough, prone to cracking", "Thin, small, light", "Underweight or low weight",
        "Light muscles, small bones", "Small, recessed, dull or lusterless", "Dry, hoarse", "Irregular, broken, or protruding",
        "Irregular, weak, tendency toward constipation", "Variable appetite and thirst", "Light, broken, prone to insomnia",
        "Cold hands/feet, poor circulation", "Quick, creative, anxious, forgetful", "Fast, restless, erratic",
        "Irregular, may experience premature ejaculation", "Warm and moist", "Anxiety, fear, loneliness",
        "Quick to learn, quick to forget", "Bursty energy, tires quickly", "Astringent foods (though needs sweet/sour/salty)"
    ],
    "Pitta": [
        "Silky, reddish or early graying, prone to thinning", "Oily, warm, may have freckles or moles", "Medium, moderate build",
        "Stable weight", "Moderate muscles", "Bright, sharp, often green/gray/copper-brown", "Sharp, commanding",
        "Sharp, yellowish, bleeding gums", "Strong digestion and metabolism", "Strong appetite and thirst", "Moderate and sound",
        "Warm body, sweaty hands and feet", "Intelligent, focused, can be irritable", "Sharp and efficient", "Intense, passionate",
        "Cool and dry", "Anger, impatience, jealousy", "Sharp and analytical", "High, sustained energy", "Spicy, acidic foods (often aggravates them)"
    ],
    "Kapha": [
        "Thick, soft, wavy, dark, and plentiful", "Soft, smooth, oily, thick, lustrous", "Large, broad, heavy",
        "Easily gains weight, hard to lose", "Strong muscles, large bones", "Large, dark, attractive, with long lashes",
        "Deep, melodious", "Large, strong, white", "Slow digestion", "Steady appetite, but slow digestion",
        "Deep and prolonged", "Moderate, stable temperature", "Calm, loyal, stable, but can be lethargic when unbalanced",
        "Slow, deliberate", "Strong and sustained without energy loss", "Warm and dry", "Attachment, greed, envy, lethargy",
        "Slow to learn, excellent long-term memory", "Slow but consistent energy", "Sweet, salty, oily foods (which often aggravate Kapha)"
    ]
}

df = pd.DataFrame(data)

# Using set_index so 'Trait' becomes the first column, giving a cleaner table display.
st.table(df.set_index("Trait"))
#endregion
