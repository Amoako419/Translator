import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit app
st.title("Text Transformation using T5 Small")

# Input text
input_text = st.text_area("Enter the text you want to transform:", height=150)

# Task selection
task = st.selectbox(
    "Select the task:",
    ["Translate to French", "Paraphrase", "Summarize"]
)

# Task prefix mapping
task_prefix_mapping = {
    "Translate to French": "translate English to French:",
    "Paraphrase": "paraphrase:",
    "Summarize": "summarize:"
}

# Transform button
if st.button("Transform"):
    if input_text.strip() == "":
        st.warning("Please enter some text to transform.")
    else:
        task_prefix = task_prefix_mapping[task]
        
        # Prepare the input text for the model
        input_text_with_prefix = f"{task_prefix} {input_text}"
        
        # Tokenize the input text
        input_ids = tokenizer.encode(input_text_with_prefix, return_tensors="pt")
        
        # Generate the output
        with st.spinner("Processing..."):
            output_ids = model.generate(input_ids)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Display the output in a text box for easy copying
        st.success("Output:")
        st.text_area("Transformed Text", value=output_text, height=150, key="output_text")