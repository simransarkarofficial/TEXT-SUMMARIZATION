import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

# Load the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model directory
model_dir = "C:\\SIMRAN\\ECA-STUDY\\PYTHON\\VSCODE PROJECTS\\TEXT SUMMARISATION\\saved_summary_model"

model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_dir)

# Clean text function
def clean_text(text):
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    text = re.sub(' +', ' ', text)  # Remove extra spaces
    text = re.sub(r'\r\n', ' ', text)  # Remove carriage and new line spaces
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.strip().lower()  # Lower case
    return text

# Define the summarize function
def summarize(text):
    text = clean_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the same device as the model

    outputs = model.generate(
        inputs["input_ids"], 
        max_length=150,  
        num_beams=4, 
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Streamlit App UI
st.title('TEXT SUMMARIZATION')
st.markdown("Hey, Its Simran, and I am going to summarize whatever you want!")

# Input text box
text_input = st.text_area("Enter text to summarize:", height=300)

# If the user clicks the 'Summarize' button
if st.button("Summarize"):
    if text_input:
        # Generate the summary
        summary = summarize(text_input)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
