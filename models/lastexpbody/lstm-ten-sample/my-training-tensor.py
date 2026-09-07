import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
def load_data(base_dir):
    all_data = []
    all_labels = []
    class_mapping = {}
    num_hands, num_landmarks = None, None

    for class_idx, class_folder in enumerate(os.listdir(base_dir)):
        class_path = os.path.join(base_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        class_mapping[class_folder] = class_idx
        samples_path = os.path.join(class_path, 'body')

        for csv_file in os.listdir(samples_path):
            if csv_file.endswith('.csv'):
                file_path = os.path.join(samples_path, csv_file)
                try:
                    df = pd.read_csv(file_path)
                    if len(df) < 920 or len(df) > 920:
                        continue

                    df = df.sort_values(by=['Frame', 'Hand', 'Landmark'])
                    pivot_table = df.pivot_table(index=['Frame', 'Hand', 'Landmark'], values=['X', 'Y', 'Z'])
                    pivot_table = pivot_table.sort_index(level=['Frame', 'Hand', 'Landmark'])
                    data = pivot_table.to_numpy()

                    num_frames = 20
                    num_hands = 2
                    num_landmarks = df['Landmark'].nunique()
                    num_samples = num_frames // 20

                    data = data.reshape((num_frames, num_hands, num_landmarks, 3))
                    reshaped_data = data.reshape((num_samples, 20, num_hands, num_landmarks, 3))
                    reshaped_data = reshaped_data.reshape((num_samples, 20, num_hands * num_landmarks * 3))

                    all_data.append(reshaped_data)
                    all_labels.extend([class_idx] * num_samples)
                except pd.errors.EmptyDataError:
                    continue

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.array(all_labels)

    print(f"Data loaded from {base_dir}:")
    print(f"Data shape: {all_data.shape}")
    print(f"Labels shape: {all_labels.shape}")

    return all_data, all_labels, class_mapping, num_hands, num_landmarks

# Load training data
train_base_dir = "../../../lastexptrain"
data, labels, class_mapping, num_hands, num_landmarks = load_data(train_base_dir)

# DEBUG!!!!
#from collections import Counter
#print(Counter(labels))


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
   #X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
# Save label encoder for future reference
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array([key for key in sorted(class_mapping.keys())])
joblib.dump(label_encoder, 'label_encoder.pkl')

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 20 * num_hands * num_landmarks * 3))
X_train_scaled = X_train_scaled.reshape(-1, 20, num_hands * num_landmarks * 3)
X_val_scaled = scaler.transform(X_val.reshape(-1, 20 * num_hands * num_landmarks * 3))
X_val_scaled = X_val_scaled.reshape(-1, 20, num_hands * num_landmarks * 3)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Save necessary components
torch.save(train_loader, 'train_loader.pth')
torch.save(val_loader, 'val_loader.pth')
joblib.dump(scaler, 'scaler.pkl')

# The rest of the code (model definition, training loop, evaluation, etc.) remains unchanged
# Use the existing sections from your original code as needed.


# Define the Transformer model
class TransformerPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(TransformerPredictor, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_layer = nn.TransformerEncoderLayer(hidden_size, nhead=6)  # Specify nhead instead of num_heads
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)  # (seq_len, batch_size, input_size)
        transformer_output = self.transformer_encoder(embedded)
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
        out = self.fc(transformer_output[:, -1, :])  # Use output from the last time step
        return out


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to create directories and files if they do not exist
def setup_results_directory():
    if not os.path.exists("results-manual-validation"):
        os.makedirs("results-manual-validation")

    existing_folders = [int(f) for f in os.listdir("results-manual-validation") if f.isdigit()]
    if existing_folders:
        new_folder = str(max(existing_folders) + 1)
    else:
        new_folder = "1"

    result_folder_path = os.path.join("results-manual-validation", new_folder)
    os.makedirs(result_folder_path)
    return result_folder_path

# Initialize the Transformer model
              #input_size = train_data.shape[2]  # Adjusted input size based on processed data shape
input_size = X_train_tensor.shape[2]  # Adjusted input size based on processed data shape

hidden_size = 510
                #num_classes = len(np.unique(train_labels))  # Number of unique classes

                #num_classes = len(np.unique(y_train_tensor.numpy()))  # Number of unique classes
num_classes = len(np.unique(y_train))  # Number of unique classes


model = TransformerPredictor(input_size, hidden_size, num_classes, num_layers=2).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print(f"Labels in training set: {np.unique(y_train)}")
print(f"Labels in validation set: {np.unique(y_val)}")
print(f"Number of classes: {num_classes}")

print("Class Mapping:", class_mapping)
print(f"Encoded Labels: {np.unique(labels)}")


# Train the Transformer model
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Evaluate the Transformer model using validation DataLoader
    val_predictions = []
    val_targets = []
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_predictions.extend(predicted.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
            val_loss = criterion(outputs, labels)  # Calculate loss here
            running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Print and save losses
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Save the trained Transformer model
torch.save(model.state_dict(), "transformer_model.pth")
print("Model saved as transformer_model.pth")

# Calculate validation metrics (precision, recall, F1-score, label accuracy)
val_precision = precision_score(val_targets, val_predictions, average='weighted')
val_recall = recall_score(val_targets, val_predictions, average='weighted')
val_f1 = f1_score(val_targets, val_predictions, average='weighted')
val_label_accuracy = accuracy_score(val_targets, val_predictions)
val_conf_matrix = confusion_matrix(val_targets, val_predictions)

# Create results directory
result_folder_path = setup_results_directory()

# Write validation results to file
with open(os.path.join(result_folder_path, "validation-results.txt"), "w") as f:
    f.write(f"Validation Precision: {val_precision:.4f}\n")
    f.write(f"Validation Recall: {val_recall:.4f}\n")
    f.write(f"Validation F1 Score: {val_f1:.4f}\n")
    f.write(f"Validation Label Accuracy: {val_label_accuracy:.4f}\n")
    f.write("\nValidation Classification Report:\n")
    f.write(classification_report(val_targets, val_predictions, zero_division=0))

# Write validation confusion matrix to file using numpy.savetxt
np.savetxt(os.path.join(result_folder_path, "validation-confusion-matrix.txt"), val_conf_matrix, fmt='%d', delimiter=',')

# Plot and save training loss graph
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.savefig(os.path.join(result_folder_path, "training_loss.png"))
plt.close()

# Plot and save confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(val_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Validation Confusion Matrix")
plt.savefig(os.path.join(result_folder_path, "validation_confusion_matrix.png"))
plt.close()

# Plot and save training and validation loss graph
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.savefig(os.path.join(result_folder_path, "training_validation_loss.png"))
plt.close()

# Print a summary of the results
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")
print(f"Validation Label Accuracy: {val_label_accuracy:.4f}")
print("\nValidation Classification Report:")
print(classification_report(val_targets, val_predictions, zero_division=0))
