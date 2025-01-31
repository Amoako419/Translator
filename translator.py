import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "google/mt5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit app
st.title("Language Translation using Google mT5")

# Input text
input_text = st.text_area("Enter the text you want to translate:")

# Language selection
target_language = st.selectbox(
    "Select the target language:",
    ["French", "German", "Spanish", "Chinese", "Japanese"]
)

# Language code mapping
language_code_mapping = {
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Chinese": "zh",
    "Japanese": "ja"
}

# Translate button
if st.button("Translate"):
    if input_text.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        target_lang_code = language_code_mapping[target_language]
        
        # Prepare the input text for the model
        input_text_with_prefix = f"translate English to {target_lang_code}: {input_text}"
        
        # Tokenize the input text
        input_ids = tokenizer.encode(input_text_with_prefix, return_tensors="pt")
        
        # Generate the translation
        with st.spinner("Translating..."):
            translated_ids = model.generate(input_ids)
            translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        
        # Display the translated text
        st.success("Translation:")
        st.write(translated_text)