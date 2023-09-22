import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

emotion_dataset = load_dataset("emotion")

emotion_df = emotion_dataset["train"].to_pandas()
features = emotion_dataset["train"].features

id2label = {label: features["label"].int2str(label) for label in range(6)}
label2id = {v: k for k, v in id2label.items()}


model_ckpt = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def tokenize_text(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=512
    )  # truncate the longest sentences


emotion_dataset = emotion_dataset.map(tokenize_text, batched=True)
class_weights = (1 - (emotion_df["label"].value_counts().sort_index() / len(emotion_df))).values

class_weights = torch.from_numpy(class_weights).float().to("cuda")

emotion_dataset = emotion_dataset.rename_column("label", "labels")


class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss_func = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_func(logits, labels)

        return (loss, outputs) if return_outputs else loss


model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=6, id2label=id2label, label2id=label2id
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    return {"f1": f1}


batch_size = 64
logging_steps = len(emotion_dataset["train"]) // batch_size
output_dir = "minilm-finetuned-emotion"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_steps=logging_steps,
    fp16=True,
)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotion_dataset["train"],
    eval_dataset=emotion_dataset["validation"],
    tokenizer=tokenizer,
)

trainer.train()
