import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from model import STAE
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
data = torch.load('stacked_tensor.pt')

# Extract input and target data
input_data = data[:, :, :-1]
target_data = data[:, 0, -1]

# Define loss function and KFold
criterion = nn.BCEWithLogitsLoss()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
batch_size = 32
epochs = 10

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs.squeeze(), targets.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def test_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred_probs = []
    all_attention_outputs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            probs = torch.sigmoid(outputs.squeeze())
            y_true.extend(targets.numpy())
            y_pred_probs.extend(probs.cpu().numpy())

            # Extract attention outputs if needed
            attention_weights = torch.softmax(model.attention(outputs), dim=1)
            attention_output = torch.sum(attention_weights * outputs, dim=1)
            all_attention_outputs.append(attention_output.cpu().numpy())

    y_pred_labels = [1 if prob > 0.5 else 0 for prob in y_pred_probs]
    return y_true, y_pred_probs, y_pred_labels, all_attention_outputs


for fold, (train_index, test_index) in enumerate(kf.split(input_data)):
    print(f"Fold {fold + 1}/5")

    X_train, X_test = input_data[train_index], input_data[test_index]
    y_train, y_test = target_data[train_index], target_data[test_index]

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = STAE(n_time_series=input_data.shape[-1], hidden_dim=32, num_layers=2, n_target=1, dropout=0.4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)

    y_true, y_pred_probs, y_pred_labels, _ = test_model(model, test_loader)

    cm = confusion_matrix(y_true, y_pred_labels)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_true, y_pred_probs)
    accuracy = accuracy_score(y_true, y_pred_labels)

    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"AUC: {auc}")
    print(f"Accuracy: {accuracy}")
