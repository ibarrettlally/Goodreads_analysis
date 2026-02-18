!pip install transformers[torch] -U
!pip install accelerate
import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import re
from sklearn.model_selection import train_test_split
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set base directory path to the project folder in Google Drive
base_directory = '/content/drive/My Drive/goodreads_project'

# Define the specific directory where GoEmotions data is stored
goemotions_directory = os.path.join(base_directory, 'data/full_dataset')

# Function to clean text data
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    return text.lower()

# Function to aggregate emotions from GoEmotions to broader categories
def aggregate_emotions(row):
    aggregated_emotions = {
        'anger': sum([row['anger'], row['annoyance'], row['disappointment']]) / 3,
        'joy': sum([row['amusement'], row['joy'], row['gratitude']]) / 3,
        'disgust': sum([row['disgust'], row['disapproval']]) / 2,
        'fear': sum([row['fear'], row['nervousness']]) / 2,
        'anticipation': sum([row['curiosity'], row['optimism'], row['excitement']]) / 3,
        'sadness': sum([row['sadness'], row['grief'], row['remorse']]) / 3,
        'surprise': row['surprise'],
        'trust': sum([row['admiration'], row['approval'], row['caring'], row['love']]) / 4
    }
    return pd.Series(aggregated_emotions)

# Function to load and prepare the GoEmotions dataset
def load_and_prepare_goemotions(directory_path, tokenizer):
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]
    all_data = pd.concat([pd.read_csv(f) for f in file_paths])
    all_data['cleaned_text'] = all_data['text'].apply(clean_text)
    tokenized_texts = tokenizer(all_data['cleaned_text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
    aggregated_emotions = all_data.apply(aggregate_emotions, axis=1)
    return tokenized_texts, aggregated_emotions.to_numpy()

# Define a custom Dataset class
class ReviewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item['labels'] = self.labels[idx]
        return item

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_checkpoint_directory = os.path.join(base_directory, 'results')
    os.makedirs(model_checkpoint_directory, exist_ok=True)  # Ensure the checkpoint directory exists

    tokenized_texts, labels = load_and_prepare_goemotions(goemotions_directory, tokenizer)
    indices = range(len(labels))
    train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)

    train_dataset = ReviewsDataset({key: tokenized_texts[key][train_indices] for key in tokenized_texts.keys()}, labels[train_indices])
    eval_dataset = ReviewsDataset({key: tokenized_texts[key][val_indices] for key in tokenized_texts.keys()}, labels[val_indices])

    training_args = TrainingArguments(
        output_dir=model_checkpoint_directory,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_dir=os.path.join(model_checkpoint_directory, 'logs'),
        save_steps=5000,
        evaluation_strategy="steps",
        eval_steps=5000,
        fp16=True,  # Enable mixed precision
    )

    trainer = Trainer(
        model=BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset  # Providing the eval_dataset for evaluation
    )

    # Start/resume training
    trainer.train()

    model_save_path = os.path.join(base_directory, 'fine_tuned_model')
    trainer.model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print("Model and tokenizer saved. Process completed.")
