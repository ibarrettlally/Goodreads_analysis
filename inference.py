import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Function to clean text data
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    return text.lower()

# Load the fine-tuned model and tokenizer
base_directory = os.path.expanduser('~/goodreads_project')
model_directory = os.path.join(base_directory, 'fine_tuned_model')
tokenizer = BertTokenizer.from_pretrained(model_directory)
model = BertForSequenceClassification.from_pretrained(model_directory, num_labels=8)

# Function to score the emotions in each text
def score_emotions(texts, tokenizer, model):
    # Tokenize the text
    tokenized_texts = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    # Get predictions from the model
    with torch.no_grad():
        outputs = model(**tokenized_texts)
    
    # Convert model outputs (logits) to probabilities
    probabilities = torch.softmax(outputs.logits, dim=1).numpy()
    return probabilities

# Process each raw CSV file, score the emotions, and save the results
for filename in os.listdir(base_directory):
    if filename.endswith("_raw.csv"):
        file_path = os.path.join(base_directory, filename)
        data = pd.read_csv(file_path)
        
        # Clean the text data
        cleaned_texts = data['review'].apply(clean_text).tolist()
        
        # Score the emotions
        emotion_scores = score_emotions(cleaned_texts, tokenizer, model)
        
        # Add the emotion scores to the DataFrame
        for i, emotion in enumerate(['anger', 'joy', 'disgust', 'fear', 'anticipation', 'sadness', 'surprise', 'trust']):
            data[emotion + '_score'] = emotion_scores[:, i]
        
        # Save the DataFrame with scores
        scored_file_path = file_path.replace("_raw.csv", "_scored.csv")
        data.to_csv(scored_file_path, index=False)

        print(f'Processed and saved: {scored_file_path}')
