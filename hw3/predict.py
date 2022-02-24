import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_path",
        default="data/public.jsonl",
        type=str,
        help="path to eval data"
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
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to output file"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    raw_data = load_dataset('json', data_files={'eval':args.eval_path})

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    eval_dataset = raw_data['eval']
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)        
    import json
    fs = open(args.output_path, 'w')
    gen_kwargs = {
        "max_length": args.max_target_length,
        "min_length": 15,
        "num_beams": 5,
        "repetition_penalty": 2.5,
        "early_stopping": True,
        "length_penalty": 1.0,
    }
    model.to('cuda')
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataset, leave=False)):
        article = batch['maintext']
        Id = batch['id']
        inputs = tokenizer(article, max_length=args.max_input_length, padding="max_length", return_tensors="pt", truncation=True).to('cuda')
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **gen_kwargs,
            )

        decoded_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        json.dump({"title":decoded_pred.replace(':', '：'), "id": Id}, fs, ensure_ascii=False)
        fs.write('\n')
    fs.close()




    


if __name__ == '__main__':
    main()    
    
