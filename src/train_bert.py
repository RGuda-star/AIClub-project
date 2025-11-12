import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    accuracy_score,

    precision_recall_fscore_support,

    classification_report,

)

from pathlib import Path
import os

from transformers import (

    BertTokenizerFast,

    BertForSequenceClassification,

    Trainer,

    TrainingArguments,

    EarlyStoppingCallback,

)
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model 


from datasets import Dataset

import torch
import torch.nn as nn 

# Check if GPU is available (if not, we'll use CPU)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

if torch.cuda.is_available():

    for i in range(torch.cuda.device_count()):

        gpu_name = torch.cuda.get_device_name(i)

        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3

        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

else:

    print("No GPU detected. Training will use CPU")

data_path = Path(__file__).resolve().parent.parent / "processed-dataset.csv" # if your processed dataset is in a data directory

df = pd.read_csv(data_path, encoding="latin-1")

df = df[["cleaned_text", "label"]].dropna()

# Convert labels to numeric (BERT expects 0 and 1)

df["label"] = df["label"].map({"OR": 1, "CG": 0})
# 80% train, 20% test

train_df, test_df = train_test_split(

    df, test_size=0.2, random_state=42, stratify=df["label"]

)

# Convert to HuggingFace Dataset format

train_ds = Dataset.from_pandas(train_df)

test_ds = Dataset.from_pandas(test_df)

print(f"Train size: {len(train_ds)}")

print(f"Test size: {len(test_ds)}")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
def tokenize(batch):

    return tokenizer(

        batch["cleaned_text"], 

        padding="max_length",    # Pad shorter reviews

        truncation=True,         # Cut off longer reviews

        max_length=512           # BERT's max length

    )

# Apply tokenization to all datasets

train_ds = train_ds.map(tokenize, batched=True)

test_ds = test_ds.map(tokenize, batched=True)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", device_map = 'auto')

# Move to CPU (if no GPU available)

model = model.to(device)
# Freeze first 6 layers, only train last 6 + classifier

for name, param in model.bert.named_parameters():

    if any(f"encoder.layer.{i}." in name for i in range(6)):

        param.requires_grad = False
def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = logits.argmax(axis=1)

    precision, recall, f1, ignore = precision_recall_fscore_support(

        labels, preds, average="binary"

    )

    acc = accuracy_score(labels, preds)

    return {

        "accuracy": acc,

        "f1": f1,

        "precision": precision,

        "recall": recall,

    }
training_args = TrainingArguments(

    output_dir="./bert_model",

    per_device_train_batch_size=8,         # Process 8 reviews at once

    per_device_eval_batch_size=8,

    num_train_epochs=3,                    # Train for 3 full passes

    learning_rate=1e-5,                    # Low learning rate for fine-tuning

    eval_strategy="epoch",                 # Evaluate after each epoch (each pass-through of the data)

    save_strategy="epoch",

    logging_steps=50,

    load_best_model_at_end=True,           # Keep best model, not last

    metric_for_best_model="eval_f1",       # Optimize for F1 score

    greater_is_better=True,

    warmup_steps=300,                      # Gradual learning rate increase

    weight_decay=0.01,                     # Regularization

    report_to=None,                        # Don't log to external services

    save_total_limit=2,                    # Only keep 2 best checkpoints

    dataloader_drop_last=True,

    remove_unused_columns=True,

)




# Clear GPU memory (if using GPU)

if torch.cuda.is_available():

    torch.cuda.empty_cache()

 # TODO: only load_in_8bit if you’re using a GPU

task_type="SEQ_CLS"  # Classification

class CastOutputToFloat(nn.Sequential):
    def forward(self,x): return super().forward(x).to(torch.float32)
model.classifier = CastOutputToFloat(model.classifier)

# Load preprocessed dataset into a dataframe (df)

df["label"] = df["label"].map({"OR": 1, "CG": 0})  # Binary labels

# Train/test split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Explicit tokenization with max_length

def tokenize_function(examples):

    return tokenizer(examples["cleaned_text"], 

                    truncation=True, padding=True, max_length=512)

# Uses DataCollatorWithPadding (for classification)

data_collator=DataCollatorWithPadding(tokenizer)

max_steps=600,              # Longer training

learning_rate=2e-5,         # Lower learning rate (10x smaller)

warmup_steps=100,

logging_steps=10,           # Less frequent logging

eval_steps=50,              # Evaluates periodically

save_steps=50,

metric_for_best_model="f1", # Optimizes for F1

# Custom metrics function

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    preds = logits.argmax(axis=1)

    precision, recall, f1, ignore = precision_recall_fscore_support(

        labels, preds, average="binary")

    # Computes accuracy, F1, precision, recall
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

config = LoraConfig(r=16, lora_alpha=32, 
                    lora_dropout=0.05,
                    bias="none",
                    task_type="SEQ_CLS") # TODO: should be the same config as the video except for the task_type (SEQ_CLS)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# TODO: assign these in the Trainer() params
trainer = Trainer(
    model=model,                          # model (should already be wrapped by get_peft_model)
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,                 # <-- evaluation dataset
    data_collator=data_collator,          # <-- collator (useful when using padding=True)
    tokenizer=tokenizer,                  # <-- helpful for evaluation/prediction pipelines
    compute_metrics=compute_metrics,      # <-- metrics function
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

eval_dataset=test_ds,

compute_metrics=compute_metrics,

# Detailed final evaluation
trainer.train()
final_results = trainer.evaluate()

print(f"Accuracy: {final_results['eval_accuracy']:.4f}")

print(f"F1 Score: {final_results['eval_f1']:.4f}")

# etc.
os.makedirs("model", exist_ok=True)
model.save_pretrained("model/bert_lora_model")
tokenizer.save_pretrained("model/bert_lora_model")
print("LoRA model saved successfully!")
folder = Path("model/bert_lora_model")
print("Files in LoRA folder:", os.listdir(folder))