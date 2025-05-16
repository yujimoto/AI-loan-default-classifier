#!/usr/bin/env python3
import torch
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys

from torch.utils.data import Dataset, random_split

import neural_network

# This class implements train/test split
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset

    def __getitem__(self, index):
        x, y = self.subset[index]
        return x, y

    def __len__(self):
        return len(self.subset)

# Test network on validation set, if it exists.
def test_network(net,testloader,print_confusion=False):
    net.eval()
    total_items = 0
    total_correct = 0
    conf_matrix = np.zeros((2,2))
    with torch.no_grad():
        for data in testloader:
            items, labels = data
            labels = labels.squeeze()
            outputs = net(items)
            _, predicted = torch.max(outputs.data, 1)
            total_items += labels.size(0)
            total_correct += torch.eq(predicted,labels).sum().item()
            conf_matrix = conf_matrix + metrics.confusion_matrix(
                labels.cpu(),predicted.cpu(),labels=[0,1])

    model_accuracy = 100.0 * total_correct / total_items
    print(', {0} test {1:.2f}%'.format(total_items,model_accuracy))
    if print_confusion:
        np.set_printoptions(precision=2, suppress=True)
        print(conf_matrix)
    net.train()

def main():
    ########################################################################
    #######                      Loading Data                        #######
    ########################################################################

    df = pd.read_csv("hw2nn_data.csv")

    X = df.iloc[:,0:16]
    y = df.iloc[:,16]

    if( neural_network.scale_inputs ):
        scale = StandardScaler()
        X = scale.fit_transform(X)
    else:
        X = X.values
        
    input = torch.tensor(X,dtype=torch.float32)
    target = torch.tensor(y.values).long()

    data = torch.utils.data.TensorDataset(input,target)

    if neural_network.train_val_split == 1:
        # Train on the entire dataset
        trainloader = torch.utils.data.DataLoader(data,
                            batch_size=neural_network.batch_size, shuffle=True)
    else:
        # Split the dataset into trainset and testset
        data.len=len(data)
        train_len = int((neural_network.train_val_split)*data.len)
        test_len = data.len - train_len
        train_subset, test_subset = random_split(data, [train_len, test_len])
        trainset = DatasetFromSubset(train_subset)
        testset = DatasetFromSubset(test_subset)

        trainloader = torch.utils.data.DataLoader(trainset,
                            batch_size=neural_network.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, 
                            batch_size=neural_network.batch_size, shuffle=False)

    # Get model, loss criterion and optimizer from neural_network]
    net = neural_network.net
    criterion = neural_network.loss_func
    optimizer = neural_network.optimizer
    # get weight initialization and lr scheduler, if appropriate
    weights_init = neural_network.weights_init
    scheduler = neural_network.scheduler

    # apply custom weight initialization, if it exists
    net.apply(weights_init)

    ########################################################################
    #######                        Training                          #######
    ########################################################################
    print("Start training...")
    for epoch in range(1,neural_network.epochs+1):
        total_loss = 0
        total_items = 0
        total_correct = 0

        for batch in trainloader:           # Load batch
            items, labels = batch 
            labels = labels.squeeze()
            preds = net(items)              # Process batch
            loss = criterion(preds, labels) # Calculate loss
            
            optimizer.zero_grad()
            loss.backward()                 # Calculate gradients
            optimizer.step()                # Update weights
            
            output = preds.argmax(dim=1)

            total_loss += loss.item()
            total_items += labels.size(0)
            total_correct += output.eq(labels).sum().item()

        # apply lr schedule, if it exists
        if scheduler is not None:
            scheduler.step()

        model_accuracy = total_correct / total_items * 100
        print('ep {0}, loss: {1:.2f}, {2} train {3:.2f}%'.format(
               epoch, total_loss, total_items, model_accuracy), end='')

        if neural_network.train_val_split < 1:
            test_network(net,testloader,
                         print_confusion=(epoch % 10 == 0))
        else:
            print()

        sys.stdout.flush()

    torch.save(net.state_dict(),'model.pth')
    print("   Model saved to model.pth")
        
if __name__ == '__main__':
    main()
