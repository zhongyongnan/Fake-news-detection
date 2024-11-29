import numpy as np
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Define the model
model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=4)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data[['Headline', 'summ', 'Stance']]
    data = data.dropna()
    label_mapping = {'disagree': 0, 'agree': 1, 'discuss': 2, 'unrelated': 3}
    data['Stance'] = data['Stance'].map(label_mapping)
    return data

def preprocess_data(data):
    def tokenize_function(examples):
        # print(examples)
        return tokenizer(examples['Headline'], examples['summ'], truncation=True, padding=True)
    
    dataset = Dataset.from_pandas(data)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("Stance", "labels")
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized_datasets

# Load the data
train_data = load_data('/data1/mwy/NLP/train_processed_article.csv')
val_data = load_data('/data1/mwy/NLP/test_processed_article.csv')

# Preprocess the data
train_dataset = preprocess_data(train_data)
val_dataset = preprocess_data(val_data)


# Debugging steps to ensure datasets are correctly prepared
print("Train dataset:", train_dataset)
# print("Validation dataset:", val_dataset)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    logging_dir='./logs',
    logging_steps=10,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Predict the results on the evaluation dataset
predictions = trainer.predict(val_dataset)

# Extract the predictions
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Save the predictions to a file
predictions_file = "./saved_roberta_model/predictions.json"
with open(predictions_file, 'w') as f:
    json.dump(predicted_labels.tolist(), f, indent=4)

print(f"Predictions saved to {predictions_file}")

# Save the model and tokenizer after training
save_directory = "./saved_roberta_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
