from datasets import Dataset, load_dataset
from transformers import GPT2Tokenizer


def preprocess_data(dataset: Dataset) -> str:
    preprocessed_data = [f"[function]: {i['function']} \n[docstring]: {i['docstring']}" for i in dataset]
    return " ".join(preprocessed_data)


def tokenize_data(text: str) -> list[int]:
    block_size = 128
    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    examples = []
    for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
        examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
    return examples


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

dataset = load_dataset("juraj-juraj/doc_gen")
sentences = preprocess_data(dataset["test"])
tokenized_sentences = tokenize_data(sentences)


from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)


def fine_tune_gpt2(model_name, train_dataset, output_dir):
    # Load GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load training dataset
    # train_dataset = TextDataset(
    #     tokenizer=tokenizer,
    #     file_path=train_file,
    #     block_size=128)
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )
    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


fine_tune_gpt2("gpt2", tokenized_sentences, "code_gpt")
