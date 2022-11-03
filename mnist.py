
# Añadimos todos los imports
import torch
import numpy as np

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import datasets

# Pre-procesamos el dataset
dataset = datasets.load_digits()
train_images = dataset.images[:1500]/255
train_target = dataset.target[:1500]
train_target = [np.eye(10)[i] for i in train_target]

test_images = dataset.images[1500:]/255
test_target = dataset.target[1500:]
test_target = [np.eye(10)[i] for i in test_target]

# Creamos el dataset personalizado
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float, device="cpu").reshape(8*8), torch.tensor(self.labels[idx], dtype=torch.float, device="cpu")
    
    def __len__(self):
        return len(self.labels)

# Definiendo los hiperparametros
learning_rate = 1.61
batch_size = 20
epochs = 100
    
# Instanciamos la clase Dataset
train_dataset = CustomDataset(train_images, train_target)
test_dataset = CustomDataset(test_images, test_target)

# Instanciamos los DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Extraemos una iteracion del dataloader
x, y = next(iter(train_dataloader))

print(f"Shape of x: {x.shape}")
print(f"Shape of y: {y.shape}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(64, 200)
        self.layer2 = nn.Linear(200, 10)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)

        return x

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        # Calculamos las predicciones y perdida
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(x)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            print(pred)
            print(y)
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

model = NeuralNetwork()
model.to("cpu")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
