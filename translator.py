import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name,legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Function to split text into chunks
def split_text_into_chunks(text, max_tokens=512):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        # Check if the current chunk exceeds the max token limit
        if len(tokenizer.encode(" ".join(current_chunk))) > max_tokens:
            # Remove the last word and add the chunk
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

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
        
        # Split the input text into chunks
        chunks = split_text_into_chunks(input_text)
        output_text = ""

        # Process each chunk
        with st.spinner("Processing..."):
            for chunk in chunks:
                # Prepare the input text for the model
                input_text_with_prefix = f"{task_prefix} {chunk}"
                
                # Tokenize the input text
                input_ids = tokenizer.encode(input_text_with_prefix, return_tensors="pt")
                
                # Generate the output
                output_ids = model.generate(input_ids)
                output_text += tokenizer.decode(output_ids[0], skip_special_tokens=True) + " "

        # Display the output in a text box for easy copying
        st.success("Output:")
        st.text_area("Transformed Text", value=output_text.strip(), height=150, key="output_text")