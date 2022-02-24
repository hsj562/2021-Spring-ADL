import argparse

from datasets import load_dataset
import torch
from tw_rouge import get_rouge
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AdamW
)
from accelerate import Accelerator
from tqdm import tqdm

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
        default="model",
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
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    raw_data = load_dataset('json', data_files={'train': args.train_path})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    accelerator = Accelerator()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_dataset = raw_data['train']

    model, optimizer = accelerator.prepare(
        model, optimizer
    )
    gen_kwargs = {
        "max_length": args.max_target_length,
        "repetition_penalty": 2.5,
        "early_stopping": True,
        "length_penalty": 1.0
    }
    # training
    model.to('cuda')
    model.train()
    for step, batch in enumerate(tqdm(train_dataset, leave=False)):
        article = batch['maintext']
        summary = batch['title']
        inputs = tokenizer(article, max_length=args.max_input_length, return_tensors="pt", truncation=True).to('cuda')
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(summary, max_length=args.max_target_length, return_tensors="pt", truncation=True).to('cuda')
        outputs = model(**inputs, labels=labels['input_ids'])        
        loss = outputs.loss
        model.eval()
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            **gen_kwargs
        )
        model.train()
        decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        rouge = get_rouge(summary, decoded_pred)
        reward = 10 * np.mean([rouge['rouge-1']['f'], rouge['rouge-2']['f'], rouge['rouge-l']['f']])
        loss = loss * reward 
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    model.save_pretrained("RL")
    tokenizer.save_pretrained("RL")
    

if __name__ == '__main__':
    main()    
    