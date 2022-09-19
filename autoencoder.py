import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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


def display_digits(model, test_loader):
    for (imgs,_) in test_loader:
        imgs = imgs.reshape(-1, 28*28)
        break
    results = model(imgs)
    k = num_epochs - 1 
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

def getRawData(number):
    transform = transforms.ToTensor()
    train_mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    idx = train_mnist_data.targets==number
    idx2 = test_mnist_data.targets==number
    
    train_mnist_data.targets = train_mnist_data.targets[idx]
    train_mnist_data.data = train_mnist_data.data[idx]
    
    test_mnist_data.targets = test_mnist_data.targets[idx2]
    test_mnist_data.data = test_mnist_data.data[idx2]
    
    data_loader = torch.utils.data.DataLoader(dataset=train_mnist_data, batch_size=64,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_mnist_data, batch_size=20)
    return data_loader, test_loader

def create_model():
    model = Autoencoder()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    return model,loss_fn,optimizer

def train_model(data_loader,model,loss_fn,optimizer):
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

def getAutoencoders(number):
    data_loader, test_loader = getRawData(number)
    model,loss_fn,optimizer = create_model()
    model = train_model(data_loader,model,loss_fn,optimizer)
    display_digits(model, test_loader)
    return model
    
def main():
    models = []
    for i in range(0,10):
        models.append(getAutoencoders(i))
    

if __name__ == '__main__':
    main()