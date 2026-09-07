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
    # Initialize lists to hold all the data and labels
    all_data = []
    all_labels = []

    # Define a mapping for class labels
    class_mapping = {}

    num_hands = None
    num_landmarks = None  # Initialize variables to store num_hands and num_landmarks

    # Loop through each class folder in the base directory
    for class_idx, class_folder in enumerate(os.listdir(base_dir)):
        class_path = os.path.join(base_dir, class_folder)

        if not os.path.isdir(class_path):
            continue  # Skip any non-directory files

        # Add the class label to the mapping
        class_mapping[class_folder] = class_idx

        # Check the 'extracted-40frames-samples-improved' folder within each class folder
        samples_path = os.path.join(class_path, 'hand')

        # Loop through each CSV file in the samples directory
        for csv_file in os.listdir(samples_path):
            if csv_file.endswith('.csv'):
                file_path = os.path.join(samples_path, csv_file)

                # Check if the file has at least 1681 rows (including the header)
                try:
                    # Load the CSV file and skip rows to get to the data part
                    df = pd.read_csv(file_path)  # Read the entire CSV file

                    # Check if the actual data part has at least 1680 rows
                    if len(df) < 840:
                        print(f"Skipping file {file_path} due to insufficient rows: {len(df)}")
                        continue
                    if len(df) > 840:
                        print(f"file {file_path} more than 1680: {len(df)}")
                        continue

                    # Sort and organize the data by Frame, Hand, and Landmark
                    df = df.sort_values(by=['Frame', 'Hand', 'Landmark'])

                    # Convert to a multi-index DataFrame for easier reshaping
                    pivot_table = df.pivot_table(index=['Frame', 'Hand', 'Landmark'], values=['X', 'Y', 'Z'])

                    # Ensure the data is sorted correctly
                    pivot_table = pivot_table.sort_index(level=['Frame', 'Hand', 'Landmark'])

                    # Convert the pivot table to a NumPy array
                    data = pivot_table.to_numpy()

                    # Assuming each frame has 42 rows (21 landmarks for each of two hands)
                    num_frames = 20  # Adjust as per your requirement
                    #num_hands = df['Hand'].nunique()
                    num_hands = 2
                    num_landmarks = df['Landmark'].nunique()
                    num_samples = num_frames // 20

                    # Reshape the data to (num_frames, 2, 21, 3)
                    data = data.reshape((num_frames, num_hands, num_landmarks, 3))

                    # Reshape the data to (num_samples, 40, 2, 21, 3)
                    reshaped_data = data.reshape((num_samples, 20, num_hands, num_landmarks, 3))

                    # Flatten the last two dimensions into one (42 * 3)
                    reshaped_data = reshaped_data.reshape((num_samples, 20, num_hands * num_landmarks * 3))

                    # Add the data and the corresponding class label to the lists
                    all_data.append(reshaped_data)
                    all_labels.extend([class_idx] * num_samples)

                except pd.errors.EmptyDataError:
                    print(f"Skipping empty file: {file_path}")
                    continue

    # Concatenate all the data and labels into single arrays
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.array(all_labels)

    print(f"Data loaded from {base_dir}:")
    print(f"Data shape: {all_data.shape}")  # Should print (total_num_samples, 40, 252)
    print(f"Labels shape: {all_labels.shape}")  # Should print (total_num_samples,)

    return all_data, all_labels, class_mapping, num_hands, num_landmarks

# Load test data
test_base_dir = "../../../lastexptest"
test_data, test_labels, _, num_hands_test, num_landmarks_test = load_data(test_base_dir)

# Load label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Convert the data for PyTorch compatibility using the loaded scaler
scaler = joblib.load('scaler.pkl')
test_data_scaled = scaler.transform(test_data.reshape(-1, 20 * num_hands_test * num_landmarks_test * 3))
test_data_scaled = test_data_scaled.reshape(-1, 20, num_hands_test * num_landmarks_test * 3)

# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(test_data_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(test_labels, dtype=torch.long)

# Create DataLoader for the test set
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16)

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
