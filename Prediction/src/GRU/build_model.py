import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from helper import train, evaluate
from data import calculate_num_features, VisitSequenceWithLabelDataset, time_collate_fn
import pickle
import os

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return self.fc(output[:, -1, :])

def evaluate_gru(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for data, labels in loader:
            data, lengths = data[0].to(device), data[1].to(device)
            labels = labels.to(device)

            outputs = model(data, lengths)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())  # Probability of the positive class

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    return accuracy, precision, recall, f1, roc_auc, mcc

def train_and_evaluate_gru():
    NUM_EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 0.005
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_seqs = pickle.load(open("./../../data/sepsis/processed_data/sepsis.seqs.train", 'rb'))
    train_labels = pickle.load(open("./../../data/sepsis/processed_data/sepsis.labels.train", 'rb'))
    valid_seqs = pickle.load(open("./../../data/sepsis/processed_data/sepsis.seqs.validation", 'rb'))
    valid_labels = pickle.load(open("./../../data/sepsis/processed_data/sepsis.labels.validation", 'rb'))

    num_features = calculate_num_features(train_seqs)

    # Create DataLoader
    train_dataset = VisitSequenceWithLabelDataset(train_seqs, train_labels)
    valid_dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels)

    weights = [1.0 if label == 1 else 1.5 for _, label in train_dataset]
    sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=time_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=time_collate_fn)

    model = GRUModel(input_size=num_features, hidden_size=64, num_layers=2, output_size=2, dropout=0.2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, DEVICE, train_loader, criterion, optimizer, epoch)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_loss, val_acc, _ = evaluate(model, DEVICE, valid_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_gru_model.pth")

    print("Training complete. Best Validation Accuracy:", best_val_acc)

    # Evaluate on the validation set
    print("\nEvaluation Metrics on Validation Set:")
    evaluate_gru(model, valid_loader, DEVICE)

# if __name__ == "__main__":
#     train_and_evaluate_gru()
