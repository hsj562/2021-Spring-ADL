import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
import pickle

token_num = 0
token_list = {}

def count_token(data):
    global token_num
    for d in data:
        words = d.split(' ')
        for w in words:
            if w not in token_list:
                token_list[w] = token_num
                token_num += 1

class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(input_dim, 30),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, output_dim),
            nn.Sigmoid()
        )


    def forward(self, x):
        out = self.seq(x)
        return out


class Data(Dataset):
    def __init__(self, ids, text, label):
        self.ids = ids
        self.text = text
        self.label = label
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        return self.ids[idx], self.text[idx], self.label[idx]

def preprocess(data):
    ret = []
    for d in data:
        bow = np.zeros(token_num, dtype=np.float16)
        for w in d.split(' '):
            if w in token_list:
                bow[token_list[w]] += 1
        ret.append(bow)
    return np.array(ret)
def test(path):
    data = pd.read_csv('test.csv', sep=',').to_numpy()
    new_data = preprocess(data[:,1])
    test_data = Data(data[:,0], new_data, data[:,2])

    test_dataloader = DataLoader(test_data, batch_size=16)
    with open('submit.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Id', 'Category'])
        for batch, (ids, text, label) in tqdm(enumerate(test_dataloader)):

            text = text.cuda()
            # print(encoded_text.size()``)
            pred = network(text.float())
            for i in range(len(ids)):
                pred[i] = 1 if pred[i] > 0.5 else 0
                writer.writerow([ids[i], int(pred[i])])
def validation(path):
    data = pd.read_csv(path, sep=',').to_numpy()
    new_data = preprocess(data[:,1])
    val_data = Data(data[:,0], new_data, data[:,2])

    val_dataloader = DataLoader(val_data, batch_size=16)

    hit = 0
    for batch, (ids, text, label) in tqdm(enumerate(val_dataloader)):

        text = text.cuda()
        label = torch.reshape(label, (label.size()[0],1)).cuda()
        pred = network(text.float())

        loss = loss_fn(pred, label.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i in range(len(ids)):
            pred[i] = 1 if pred[i] > 0.5 else 0
            if pred[i] == label[i]:
                hit += 1
    print(f'acc: {hit/val_data.__len__()}')
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load train data
    train_data = pd.read_csv('train.csv', sep=',').to_numpy()
    
    # load dev data
    # dev_data = pd.read_csv('dev.csv', sep=',')
    # train_data.append(dev_data)

    count_token(train_data[:,1])
    new_text = preprocess(train_data[:,1])

    data = Data(train_data[:,0], new_text, train_data[:,2])
    train_dataloader = DataLoader(data, batch_size=16)
    # tokenize
    network = NN(token_num, 1) 
    network.to(device)

    loss_fn = nn.BCELoss()
    learning_rate = 0.003
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    print('start to train')
    # train

    for epoch in tqdm(range(10)):
        for batch, (ids, text, label) in tqdm(enumerate(train_dataloader)):
            text = text.cuda()
            # print(encoded_text.size())
            pred = network(text.float())
            
            label = torch.reshape(label, (label.size()[0],1)).cuda()
            # print(pred.size(), label.size()[0])
            loss = loss_fn(pred, label.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss)

    # validation
    
    validation('dev.csv')
    
    # pickle.dump(network, open('model.pkl', 'wb'))
    
    # with open('model.pkl', 'rb') as f:
    #     network = pickle.load(f)
    # test

    test('test.csv')

