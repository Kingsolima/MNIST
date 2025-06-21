import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN
import matplotlib.pyplot as plt

# Constants
EPOCHS = 6
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LEARNING_RATE = 0.001
BREAK_ACCURACY = 0.995

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=TEST_BATCH_SIZE, shuffle=False
)

# Init model, loss, optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = RMSprop(model.parameters(), lr=LEARNING_RATE)

# Training function
def train(model, loader):
    model.train()
    train_loss, train_acc = [], []
    for epoch in range(EPOCHS):
        correct, total, running_loss = 0, 0, 0.0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        acc = correct / total
        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
        train_loss.append(avg_loss)
        train_acc.append(acc)
        if acc >= BREAK_ACCURACY:
            print("Reached target accuracy, stopping early.")
            break
    return train_loss, train_acc

# Evaluation function
def test(model, loader):
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    avg_loss = loss / len(loader)
    acc = correct / len(loader.dataset)
    print(f"Test: Loss={avg_loss:.4f}, Accuracy={acc*100:.2f}%")

# Run training and testing
train_loss, train_acc = train(model, train_loader)
test(model, test_loader)

# Plotting
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Loss')
plt.title("Training Loss")
plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Accuracy')
plt.title("Training Accuracy")
plt.tight_layout()
plt.show()
