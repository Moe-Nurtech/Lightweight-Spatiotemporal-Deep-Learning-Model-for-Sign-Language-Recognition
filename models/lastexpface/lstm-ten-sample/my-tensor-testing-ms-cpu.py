import time
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
    num_hands = None
    num_landmarks = None

    for class_idx, class_folder in enumerate(os.listdir(base_dir)):
        class_path = os.path.join(base_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        class_mapping[class_folder] = class_idx
        samples_path = os.path.join(class_path, 'extracted-40frames-samples-improved')

        for csv_file in os.listdir(samples_path):
            if csv_file.endswith('.csv'):
                file_path = os.path.join(samples_path, csv_file)
                try:
                    df = pd.read_csv(file_path)
                    if len(df) < 840:
                        print(f"Skipping file {file_path} due to insufficient rows: {len(df)}")
                        continue
                    if len(df) > 840:
                        print(f"file {file_path} more than 840: {len(df)}")
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
                    print(f"Skipping empty file: {file_path}")
                    continue

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.array(all_labels)

    print(f"Data loaded from {base_dir}:")
    print(f"Data shape: {all_data.shape}")
    print(f"Labels shape: {all_labels.shape}")

    return all_data, all_labels, class_mapping, num_hands, num_landmarks


# Load test data
test_base_dir = "../../../test"
test_data, test_labels, _, num_hands_test, num_landmarks_test = load_data(test_base_dir)

# Load scaler and label encoder
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Scale test data
test_data_scaled = scaler.transform(test_data.reshape(-1, 20 * num_hands_test * num_landmarks_test * 3))
test_data_scaled = test_data_scaled.reshape(-1, 20, num_hands_test * num_landmarks_test * 3)

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(test_data_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(test_labels, dtype=torch.long)

# Create DataLoader
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16)


class TransformerPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(TransformerPredictor, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_layer = nn.TransformerEncoderLayer(hidden_size, nhead=6)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)
        transformer_output = self.transformer_encoder(embedded)
        transformer_output = transformer_output.permute(1, 0, 2)
        out = self.fc(transformer_output[:, -1, :])
        return out


def setup_test_results_directory():
    if not os.path.exists("test-results"):
        os.makedirs("test-results")
    return "test-results"


# Initialize model on CPU
input_size = test_data.shape[2]
hidden_size = 510
num_classes = len(label_encoder.classes_)
model = TransformerPredictor(input_size, hidden_size, num_classes, num_layers=2)
model.load_state_dict(torch.load("transformer_model.pth", map_location=torch.device("cpu")))

# Evaluate
model.eval()
test_predictions = []
test_targets = []
total_samples = 0
start_time = time.time()

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_predictions.extend(predicted.numpy())
        test_targets.extend(labels.numpy())
        total_samples += inputs.size(0)

end_time = time.time()
avg_time_per_sample_ms = (end_time - start_time) / total_samples * 1000

print(f"Average execution time per sample: {avg_time_per_sample_ms:.2f} ms")

# Metrics
test_precision = precision_score(test_targets, test_predictions, average='weighted', zero_division=0)
test_recall = recall_score(test_targets, test_predictions, average='weighted', zero_division=0)
test_f1 = f1_score(test_targets, test_predictions, average='weighted', zero_division=0)
test_label_accuracy = accuracy_score(test_targets, test_predictions)
test_conf_matrix = confusion_matrix(test_targets, test_predictions)

# Save results
test_result_folder_path = setup_test_results_directory()
with open(os.path.join(test_result_folder_path, "test-results.txt"), "w") as f:
    f.write(f"Test Precision: {test_precision:.4f}\n")
    f.write(f"Test Recall: {test_recall:.4f}\n")
    f.write(f"Test F1 Score: {test_f1:.4f}\n")
    f.write(f"Test Label Accuracy: {test_label_accuracy:.4f}\n")
    f.write(f"Average Execution Time per Sample: {avg_time_per_sample_ms:.2f} ms\n")
    f.write("\nTest Classification Report:\n")
    f.write(classification_report(test_targets, test_predictions, zero_division=0))

np.savetxt(os.path.join(test_result_folder_path, "test-confusion-matrix.txt"), test_conf_matrix, fmt='%d', delimiter=',')

plt.figure(figsize=(10, 7))
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.savefig(os.path.join(test_result_folder_path, "test_confusion_matrix.png"))
plt.close()
