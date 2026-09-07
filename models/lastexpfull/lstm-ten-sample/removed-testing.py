import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(base_dir):
    all_data = []
    all_labels = []
    class_mapping = {}

    for class_idx, class_folder in enumerate(os.listdir(base_dir)):
        class_path = os.path.join(base_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        class_mapping[class_folder] = class_idx
        samples_path = os.path.join(class_path, 'extracted-20frames-samples-improved-full')

        for csv_file in os.listdir(samples_path):
            if not csv_file.endswith('.csv'):
                continue

            file_path = os.path.join(samples_path, csv_file)
            try:
                df = pd.read_csv(file_path)
                if df['Frame'].nunique() != 20:
                    continue  # ensure 20 frames per sample

                df = df.sort_values(by=['Frame', 'Category', 'Landmark_Index'])

                # Flatten each frame to a 1D vector (preserve all rows)
                frame_vectors = []
                for frame_id in sorted(df['Frame'].unique()):
                    frame_df = df[df['Frame'] == frame_id]
                    frame_values = frame_df[['X', 'Y', 'Z']].values.flatten()

                    # Pad or truncate frame to 153 features
                    max_features_per_frame = 153
                    if len(frame_values) < max_features_per_frame:
                        padded = np.pad(frame_values, (0, max_features_per_frame - len(frame_values)))
                    else:
                        padded = frame_values[:max_features_per_frame]

                    frame_vectors.append(padded)

                sample_array = np.stack(frame_vectors)  # shape: (20, 153)
                all_data.append(sample_array)
                all_labels.append(class_idx)

            except Exception as e:
                print(f"Skipping {csv_file}: {e}")
                continue

    all_data = np.stack(all_data)  # (num_samples, 20, 153)
    all_labels = np.array(all_labels)

    print(f"Loaded {len(all_data)} samples from {base_dir}")
    print(f"Data shape: {all_data.shape}")
    print(f"Labels shape: {all_labels.shape}")
    return all_data, all_labels, class_mapping


# --- LOAD TEST DATA ---
test_base_dir = "../../../test-hand-body-face"
test_data, test_labels, test_class_mapping = load_data(test_base_dir)

# --- LOAD label_remap and APPLY TO test_data and test_labels ---
label_remap = joblib.load("label_remap.pkl")  # your remapping dict {old_label: new_label}

# Filter valid samples whose labels are in remap keys
valid_indices = [i for i, lbl in enumerate(test_labels) if lbl in label_remap]
test_data = test_data[valid_indices]
test_labels = np.array([label_remap[lbl] for lbl in test_labels[valid_indices]])

# --- LOAD label encoder with remapped classes ---
label_encoder = joblib.load("label_encoder.pkl")

# --- SCALE test data ---
scaler = joblib.load("scaler.pkl")
num_features = test_data.shape[1] * test_data.shape[2]  # 20*153=3060
test_data_scaled = scaler.transform(test_data.reshape(-1, num_features))
test_data_scaled = test_data_scaled.reshape(-1, test_data.shape[1], test_data.shape[2])

# --- Convert to torch tensors ---
X_test_tensor = torch.tensor(test_data_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(test_labels, dtype=torch.long)

# --- Create DataLoader ---
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the LSTM model class
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

# Function to create test results directory
def setup_test_results_directory():
    if not os.path.exists("test-results"):
        os.makedirs("test-results")
    return "test-results"


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the trained LSTM model
input_size = test_data.shape[2]
hidden_size = 510
num_classes = len(label_encoder.classes_)


model = TransformerPredictor(input_size, hidden_size, num_classes, num_layers=2).to(device)


model.load_state_dict(torch.load("transformer_model.pth"))

# Evaluate the LSTM model using the test DataLoader
test_predictions = []
test_targets = []
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        #outputs = model(inputs.unsqueeze(1))
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_predictions.extend(predicted.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

# Calculate test metrics (precision, recall, F1-score, label accuracy)
test_precision = precision_score(test_targets, test_predictions, average='weighted', zero_division=0)
test_recall = recall_score(test_targets, test_predictions, average='weighted', zero_division=0)
test_f1 = f1_score(test_targets, test_predictions, average='weighted', zero_division=0)
test_label_accuracy = accuracy_score(test_targets, test_predictions)
test_conf_matrix = confusion_matrix(test_targets, test_predictions)

# Create test results directory
test_result_folder_path = setup_test_results_directory()

# Write test results to file
with open(os.path.join(test_result_folder_path, "test-results.txt"), "w") as f:
    f.write(f"Test Precision: {test_precision:.4f}\n")
    f.write(f"Test Recall: {test_recall:.4f}\n")
    f.write(f"Test F1 Score: {test_f1:.4f}\n")
    f.write(f"Test Label Accuracy: {test_label_accuracy:.4f}\n")
    f.write("\nTest Classification Report:\n")
    f.write(classification_report(test_targets, test_predictions, zero_division=0))

# Write test confusion matrix to file using numpy.savetxt
np.savetxt(os.path.join(test_result_folder_path, "test-confusion-matrix.txt"), test_conf_matrix, fmt='%d', delimiter=',')

# Plot and save confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.savefig(os.path.join(test_result_folder_path, "test_confusion_matrix.png"))
plt.close()
