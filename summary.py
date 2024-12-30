import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model directory
model_dir = "C:\SIMRAN\ECA-STUDY\PYTHON\VSCODE PROJECTS\TEXT SUMMARISATION\saved_summary_model"

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_dir)

# Clean text function (if you used one in Colab)
import re 

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

    # Remove special characters
  # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove extra spaces
  text = re.sub(' +', ' ', text)

    # Remove carriage and new line spaces
  text = re.sub(r'\r\n',' ',text)

    # Remove HTML tags
  text = re.sub(r'<.*?>','',text)

    # Lower Case
  text = text.strip().lower()

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

# Test with a sample input
sample_dialogue = """
Alex: Hey Sam! I just came across an amazing podcast about space exploration. Thought you'd love it!
Alex: It's all about the latest missions to Mars and what we could find there in the future.
Sam: Oh, that sounds so exciting! I’ve been reading a lot about Mars too. The idea of human missions there is incredible.
Alex: Right? They also talk about the technology needed for colonization, which is mind-blowing.
Sam: That's fascinating! I’ve been following the progress of the Perseverance rover. It’s like we’re watching history unfold.
Alex: I totally agree! And the podcast also discussed new advancements in space suits. So cool!
Sam: I need to check it out. Maybe I'll listen to it on my way to work tomorrow. 
Alex: Definitely, you should! I’ll send you the link.
Sam: Thanks, Alex! Can’t wait to dive into it!
"""

summary = summarize(sample_dialogue)
print("Summary:", summary)
