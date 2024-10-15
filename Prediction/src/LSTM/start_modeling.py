import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.optim as optim

from helper import train, evaluate, best_evaluate
from data import calculate_num_features, VisitSequenceWithLabelDataset, time_collate_fn
from build_model import LSTM_Model


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, seqs):
        # Dynamically compute lengths for the current batch of sequences
        current_batch_size = seqs.size(0)
        
        # Generate lengths assuming all sequences are the same length, or modify based on actual dataset logic
        lengths_batch = torch.full((current_batch_size,), fill_value=seqs.size(1), dtype=torch.long).to(seqs.device)
        
        # Pass both the sequences and the corresponding lengths to the model
        return self.model((seqs, lengths_batch))



torch.manual_seed(0)

PATH_TRAIN_SEQS = "./../../data/sepsis/processed_data/sepsis.seqs.train"
PATH_TRAIN_LABELS = "./../../data/sepsis/processed_data/sepsis.labels.train"
PATH_VALID_SEQS = "./../../data/sepsis/processed_data/sepsis.seqs.validation"
PATH_VALID_LABELS = "./../../data/sepsis/processed_data/sepsis.labels.validation"
PATH_TEST_SEQS = "./../../data/sepsis/processed_data/sepsis.seqs.test"
PATH_TEST_LABELS = "./../../data/sepsis/processed_data/sepsis.labels.test"
PATH_OUTPUT = "./../../out/selected_model_lstm/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

NUM_EPOCHS = 20
BATCH_SIZE = 64
NUM_WORKERS = 0

# Data loading
print('===> Loading entire datasets')
train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
train_labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))

num_features = calculate_num_features(train_seqs)

train_dataset = VisitSequenceWithLabelDataset(train_seqs, train_labels)
valid_dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels)
test_dataset = VisitSequenceWithLabelDataset(test_seqs, test_labels)

count_positive = 0
count_negative = 0
for data, label in train_dataset:
    if label == 1:
        count_positive += 1
    else:
        count_negative += 1

weights = [count_negative if label == 1 else count_positive for data, label in train_dataset]
sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, collate_fn=time_collate_fn, sampler=sampler, num_workers=NUM_WORKERS)

valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=time_collate_fn,
                          num_workers=NUM_WORKERS)

test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=time_collate_fn,
                         num_workers=NUM_WORKERS)

model = LSTM_Model(num_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

device = torch.device("cpu")
model.to(device)
criterion.to(device)

best_val_acc = 0.0
for epoch in range(NUM_EPOCHS):
    train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
    valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

    is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
    if is_best:
        best_val_acc = valid_accuracy
        torch.save(model, os.path.join(PATH_OUTPUT, "LSTM_Model.pth"))

best_model = torch.load(os.path.join(PATH_OUTPUT, "LSTM_Model.pth"))

print("\nEvaluation metrics on train set: \t")
best_evaluate(best_model, device, train_loader)

print("\nEvaluation metrics on validation set: \t")
best_evaluate(best_model, device, valid_loader)

print("\nEvaluation metrics on test set: \t")
best_evaluate(best_model, device, test_loader)

# Function to calculate permutation feature importance
# Function to calculate permutation feature importance with multiple iterations
def permutation_feature_importance(model, loader, device, criterion, num_features, num_iterations=10):
    model.eval()
    
    # Evaluate model on original dataset
    base_accuracy = evaluate(model, device, loader, criterion)[1]
    print(f"Base Accuracy: {base_accuracy}")

    feature_importances = np.zeros(num_features)
    
    for feature_idx in range(num_features):
        print(f"Permuting feature {feature_idx + 1}/{num_features}")
        permuted_accuracies = []
        
        for iteration in range(num_iterations):
            accuracies = []
            
            for i, (data, target) in enumerate(loader):
                seqs, lengths = data  # Assuming `data` is a tuple of (sequences, lengths)
                
                seqs = seqs.to(device)
                lengths = lengths.to(device)
                target = target.to(device)
                
                # Save original data
                original_data = seqs.clone()
                
                # Permute the current feature across all batches
                permuted_data = original_data.clone()
                permuted_data[:, :, feature_idx] = permuted_data[torch.randperm(permuted_data.size(0)), :, feature_idx]
                
                # Evaluate model on permuted data
                output = model((permuted_data, lengths))
                pred = output.argmax(dim=1, keepdim=True)
                accuracies.append(accuracy_score(target.cpu().numpy(), pred.cpu().numpy()))

            # Average accuracy for this permutation iteration
            permuted_accuracy = np.mean(accuracies)
            permuted_accuracies.append(permuted_accuracy)

        # Calculate importance as the average drop in accuracy over all iterations
        avg_permuted_accuracy = np.mean(permuted_accuracies)
        importance = base_accuracy - avg_permuted_accuracy
        feature_importances[feature_idx] = importance

    return feature_importances


importances = permutation_feature_importance(best_model, test_loader, device, criterion, num_features, num_iterations=10)

# Plot the updated feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(num_features), importances, color="r", align="center")
plt.xlabel("Feature Index")
plt.ylabel("Importance (Accuracy Drop)")
plt.title("Permutation Feature Importance for LSTM Model (Averaged over 10 iterations)")
plt.savefig(os.path.join(PATH_OUTPUT, "lstm_feature_importance.png"))




