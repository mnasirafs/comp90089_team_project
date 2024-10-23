import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix


# Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Modified Transformer Model (Only Encoder)
class TransformerEncoderModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerEncoderModel, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # Only Encoder Part
        return self.fc_out(x[:, -1, :])  # Output the last time step's output

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

# Training Loop
def train_transformer(model, train_loader, criterion, optimizer, device):
    model.train()
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation Loop
def evaluate_transformer(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device).view(-1)  # Flatten labels

            outputs = model(data)
            probs = torch.sigmoid(outputs).view(-1)  # Flatten probabilities

            # Store predictions and probabilities
            y_true.extend(labels.cpu().numpy())
            y_pred.extend((probs >= 0.5).cpu().numpy())  # Threshold for binary classification
            y_prob.extend(probs.cpu().numpy())

    # Convert lists to numpy arrays and ensure correct shapes
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_prob = np.array(y_prob).flatten()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print all metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'mcc': mcc,
        'conf_matrix': conf_matrix
    }


# Load Data from CSV/NumPy
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Ensure all values are numeric and fill NaNs
    train_df = train_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    test_df = test_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Extract features and labels
    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.float32)

    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.float32)

    return X_train, y_train, X_test, y_test

# Main Function to Run the Transformer
def run_transformer():
    input_size = 12  # Number of features
    output_size = 1  # Regression (for binary classification use 1 and BCEWithLogitsLoss)
    d_model = 64
    nhead = 4
    num_encoder_layers = 3
    dim_feedforward = 128
    dropout = 0.2
    learning_rate = 1e-3
    num_epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the datasets
    X_train, y_train, X_test, y_test = load_data(
        "../../data/sepsis/train/train_sample_cleaned_pivoted_vital.csv", 
        "../../data/sepsis/test/test_sample_cleaned_pivoted_vital.csv"
    )

    # Create Dataset and DataLoader
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize the model, loss function, and optimizer
    model = TransformerEncoderModel(input_size, output_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout).to(device)
    criterion = nn.MSELoss()  # Use nn.BCEWithLogitsLoss() for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and Evaluation Loop
    for epoch in range(num_epochs):
        train_transformer(model, train_loader, criterion, optimizer, device)
      
        print(f"Epoch {epoch + 1}/{num_epochs} completed.")
        # Evaluate the model on the test set
    metrics = evaluate_transformer(model, test_loader, device)

    # Save metrics for comparison with other models (e.g., GRU, LSTM)
    print("Metrics saved for comparison.",metrics)


if __name__ == "__main__":
    run_transformer()  