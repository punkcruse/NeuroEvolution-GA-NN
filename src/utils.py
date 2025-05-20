
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score

def train_and_evaluate(model, train_loader, val_loader, lr, optimizer_name='adam'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) if optimizer_name == 'adam' else optim.SGD(model.parameters(), lr=lr)

    for _ in range(2):  # Train for 2 epochs
        model.train()
        for x, y in train_loader:
            x, y = x.view(x.size(0), -1).to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.view(x.size(0), -1).to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return accuracy, precision, recall, f1

def save_checkpoint(model, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
