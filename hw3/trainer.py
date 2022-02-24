import argparse

from datasets import load_dataset
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    set_seed,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        default="data/train.jsonl",
        type=str,
        help="path to train data"
    )
    parser.add_argument(
        "--model",
        default="google/mt5-small",
        type=str,
        help="model type",
    )
    parser.add_argument(
        "--max_input_length",
        default=1024,
        type=int,
        help="max length of input"
    )
    parser.add_argument(
        "--max_target_length",
        default=128,
        type=int,
        help="max length of target"
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="batch size"
    )
    parser.add_argument(
        "--gradient_accumulation",
        default=16,
        type=int,
        help="gradient accumulation"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    raw_data = load_dataset('json', data_files={'train': args.train_path})

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    def preprocess(example):
        maintext = example['maintext']
        summary = example['title']
        inputs = tokenizer(maintext, max_length=args.max_input_length, padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(summary, max_length=args.max_target_length, padding="max_length", truncation=True)
        inputs['labels'] = labels['input_ids']
        return inputs
    
    train_dataset = raw_data['train'].map(preprocess, batched=True, remove_columns=['date_publish', 'maintext', 'source_domain', 'split', 'title'])
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer_args = Seq2SeqTrainingArguments(
        "checkpoints",
        save_strategy = "epoch",
        report_to="none",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        weight_decay=0.01,
        num_train_epochs=20,
        predict_with_generate=True,
        gradient_accumulation_steps=args.gradient_accumulation,
    )
    trainer = Seq2SeqTrainer(
        model,
        trainer_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == '__main__':
    main()    
    