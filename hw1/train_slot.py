import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange
import torch.nn as nn

from dataset import SeqClsDataset
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
EARLY_STOP = 10

def cal_acc(outputs, labels, token_len):
    cnt = 0
    for idx in range(len(token_len)):
        # print(outputs[idx, :token_len[idx]])
        res = torch.argmax(outputs[idx, :token_len[idx]], 1).tolist()
        label = labels[idx, :token_len[idx]].tolist()
        cnt += 1 if res == label else 0
    return cnt

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    from torch.utils.data import DataLoader
    import utils
    
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, collate_fn=datasets['train'].collate_fn, shuffle=True, num_workers=4)
    val_loader = DataLoader(datasets['eval'], batch_size=args.batch_size, collate_fn=datasets['eval'].collate_fn, shuffle=False, num_workers=4)
    # TODO: crecate DataLoader for train / dev datasets

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    # TODO: init model and move model to target device(cpu / gpu)
    device = 'cuda'
    
    from model import SeqClassifier
    from tqdm import tqdm

    model = SeqClassifier(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional, num_class=len(tag2idx)).to(device)
    
    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    stop_cnt = 0
    best_acc = 0.0
    best_loss = 100.0
    model_path = './ckpt/slot/model.ckpt'
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        train_acc = 0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        
        # Training
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data['data'].to(device), data['label'].to(device)
            token_len = data['token_len']
            optimizer.zero_grad()
            # print(inputs.shape)
            outputs = model(inputs)
            train_acc += cal_acc(outputs, labels, token_len)
            dim = labels.shape[0] * labels.shape[1]

            labels = torch.reshape(labels, (dim, ))
            outputs = torch.reshape(outputs, (dim, 9))
            batch_loss = criterion(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                inputs, labels = data['data'].to(device), data['label'].to(device)
                token_len = data['token_len']
                outputs = model(inputs)
                batch_loss = 0.0
                
                val_acc += cal_acc(outputs, labels, token_len)
                
                dim = labels.shape[0] * labels.shape[1]
                labels = torch.reshape(labels, (dim, ))
                outputs = torch.reshape(outputs, (dim, 9))
                batch_loss = criterion(outputs, labels)
                
                _, val_pred = torch.max(outputs, 1)

                val_loss += batch_loss.item()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        # print(f'val_loss: {val_loss}')
        print(f"[{epoch}/{args.num_epoch}] Train Acc: {train_acc/len(datasets['train'])} Train Loss: {train_loss}, Val Acc: {val_acc/len(datasets['eval'])} Val Loss: {val_loss}")
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"save model with acc {val_acc/len(datasets['eval'])}")
            stop_cnt = 0
        else:
            stop_cnt += 1
        
        if stop_cnt >= EARLY_STOP:
            print('early stopping')
            break

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=7e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=50)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
