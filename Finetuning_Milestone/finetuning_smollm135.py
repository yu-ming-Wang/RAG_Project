from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from clearml import Task
import pandas as pd

task = Task.init(project_name="FineTuning_Pipeline", task_name="Smollm135_Finetuning")
# Hugging Face authentication token
huggingface_token = "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"
model_name = "HuggingFaceTB/SmolLM2-135M"

# Load pre-trained model and tokenizer with authentication
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=huggingface_token)

# Define LoRA configuration
lora_config = LoraConfig(
    task_type="CAUSAL_LM",        # Task type
    r=64,                          # Low-rank dimension
    lora_alpha=16,                # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.1,             # Dropout probability
    bias="none"
)
task.connect(lora_config)
# Apply LoRA to the model
model = get_peft_model(model, lora_config)


output_dir = "./results"
output_log_dir = "./log"

dataset = load_dataset("csv", data_files={"train": "train_data.csv", "test": "test_data.csv"}, delimiter=",")

tokenizer.pad_token = tokenizer.eos_token

# Preprocessing the dataset
def preprocess_data(example):
    inputs = tokenizer(
        example["prompt"],  # Replace "prompt" with your dataset's input column name
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    labels = tokenizer(
        example["answer"],  # Replace "text" with your dataset's output column name
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )["input_ids"]
    inputs["labels"] = labels
    return inputs

tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir=output_log_dir,
    learning_rate=1e-4,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
)
task.connect(training_args)

# Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],  # Replace with your eval dataset if available
)

# Start fine-tuning
trainer.train()

model.save_pretrained(output_dir)

# Upload trained model directory to ClearML Artifact Store
task.upload_artifact("fine_tuned_model", output_dir)

# Upload tokenized dataset if needed
task.upload_artifact("tokenized_dataset", "train_data.csv")
task.upload_artifact("tokenized_dataset", "test_data.csv")

# Optionally upload log information
task.upload_artifact("training_logs", output_log_dir)

# Finish the ClearML Task
task.close()