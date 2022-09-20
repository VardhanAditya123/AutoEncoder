from cProfile import label
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix

IMG_SIZE = 28 * 28
sze = 0
num_epochs = 5

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Linear(IMG_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, IMG_SIZE),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

def getRawData():
    transform = transforms.ToTensor()
    test_mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_mnist_data)
    return test_loader

def loadModels():
    PATH = "model.pt"
    mdict = torch.load(PATH)
    return mdict

def runEncoder(models, img):
    loss_fn = nn.MSELoss();
    lossDict = {}
    for digit,model in models.items():
        img = img.reshape(-1, 28*28)
        outputs = model(img)
        loss = loss_fn(outputs, img)
        lossDict[digit] = loss.item()
    return min(lossDict, key=lossDict.get)   
    
def runModel(models,testLoader):
    preds = []
    labels = []
    for (img,label) in testLoader:
        pred = runEncoder(models,img)
        preds.append(pred)
        labels.append(label.item())
    confMatrix(preds, labels)
   
 
def confMatrix(preds, labelValues):
    
    mlabels = [0,1,2,3,4,5,6,7,8,9]
    confusion = metrics.confusion_matrix(labelValues, preds, labels=mlabels)
    report = metrics.classification_report(labelValues, preds, labels=mlabels)
    print("\nConfusion Matrix:\n")
    print(confusion)
    print("\nReport:")
    print(report)
       
def main():
    testLoader = getRawData()
    mdict = loadModels()
    runModel(mdict,testLoader)
    return mdict
    

if __name__ == '__main__':
    main()