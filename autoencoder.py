from cgi import test
from unicodedata import name
from unittest import result
from numpy import ma
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
train_mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=train_mnist_data, batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_mnist_data, batch_size=20)

IMG_SIZE = 28 * 28
sze = 0
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

def create_model():
    model = Autoencoder()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    return model,loss_fn,optimizer

def train_model():
    model,loss_fn,optimizer = create_model()
    num_epochs = 20
    results = []
    for epoch in range(num_epochs):
        for (img, x) in data_loader:
            # zero the parameter gradients
            optimizer.zero_grad()

            img = img.reshape(-1, 28*28)
        
            # predict classes using images from the training set
            outputs = model(img)
            
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, img)
            
            # backpropagate the loss
            loss.backward()

            # adjust parameters based on the calculated gradients
            optimizer.step()

        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        results.append((img,outputs))
    return model

def display_digits():
    model = train_model()
    for (imgs,_) in test_loader:
        imgs = imgs.reshape(-1, 28*28)
        break
    
    results = model(imgs)
    k = 19
    sze = results.size(0)
    plt.figure()
    plt.gray()
    outputs = results.detach().numpy()
    c = 1
    for i, item in enumerate(imgs):
        plt.subplot(2, sze, c)
        c = c + 1
        item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        plt.imshow(item[0])
            
    for i, item in enumerate(outputs):
        plt.subplot(2, sze, c) # row_length + i + 1
        c = c + 1
        item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        plt.imshow(item[0])
    plt.show()

def main():
    display_digits()

if __name__ == '__main__':
    main()