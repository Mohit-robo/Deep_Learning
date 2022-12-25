import os
import json
import time
import torch 
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet
from nltk_utils import stem, tokenize, bag_of_words

with open('intents.json', 'r') as f:
    intents = json.load(f)

## tokenizing the data
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

## stemming
ignore_words = ['?','!','.',';',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]

## keeping only unique words and tags
all_words = sorted(set(all_words))
tags = sorted(set(tags))

## implementing bag of words
x_train = []
y_train = []
for (patter_sent, tag) in xy:
    bag = bag_of_words(patter_sent, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

## Datapoints
x_train = np.array(x_train)
y_train = np.array(y_train)

## Creating dataset class
class ChatDataset(Dataset):
    def __init__(self) :
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index) :
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

## Hyperparams
### dataset hyperparams

batch_size = 4
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)

##general
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001
num_epochs = 500

## Loading Data
dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

## Creating the training loop

### Loading model 
model = NeuralNet(input_size, hidden_size, output_size).to(device)

### Loss and Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = lr) 
loss_fun = nn.CrossEntropyLoss()

### Main training loop
def train(train_loader,model,loss_fun,optimizer,num_epochs,FILE = 'weights/data.pth'):
    for epoch in range(num_epochs):

        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)

            #forward pass
            outputs = model(words)
            loss = loss_fun(outputs, labels)

            ##backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch +1) % 100 == 0:
            print(f'epoch {epoch +1}/{num_epochs}, loss={loss.item():.4f}')

    print(f'final loss, loss = {loss.item():.4f}')

    data = {

        "model_state": model.state_dict(),
        "input_size":input_size,
        "output_size":output_size,
        "hidden_size":hidden_size,
        "all_words":all_words,
        "tags":tags
    }

    torch.save(data, FILE)

    print(f'Training complete, file save to {os.path.abspath(FILE)} ')

start = time.time()
train(train_loader,model,loss_fun,optimizer,num_epochs)

print(f'Time taken for training : {time.time() - start} ')