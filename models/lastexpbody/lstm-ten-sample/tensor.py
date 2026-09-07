import pandas as pd
import numpy as np
import os

# Load the CSV file
file_path = '../../../train/0001/extracted-40frames-samples-improved/14-1-selected.csv'
df = pd.read_csv(file_path)

# Extract the class label from the file path (assuming the directory name is the class label)
# Split the path and find the directory name 'accident'
class_label = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

# Sort and organize the data by Frame, Hand, and Landmark
df = df.sort_values(by=['Frame', 'Hand', 'Landmark'])

# Convert to a multi-index DataFrame for easier reshaping
pivot_table = df.pivot_table(index=['Frame', 'Hand', 'Landmark'], values=['X', 'Y', 'Z'])

# Ensure the data is sorted correctly
pivot_table = pivot_table.sort_index(level=['Frame', 'Hand', 'Landmark'])

# Convert the pivot table to a NumPy array
data = pivot_table.to_numpy()

# Assuming each frame has 42 rows (21 landmarks for each of two hands)
num_frames = df['Frame'].nunique()
num_hands = df['Hand'].nunique()
num_landmarks = df['Landmark'].nunique()
num_samples = num_frames // 20

# Reshape the data to (num_frames, 2, 21, 3)
data = data.reshape((num_frames, num_hands, num_landmarks, 3))

# Reshape the data to (num_samples, 40, 2, 21, 3)
reshaped_data = data.reshape((num_samples, 20, num_hands, num_landmarks, 3))

# Flatten the last two dimensions into one (42 * 3)
reshaped_data = reshaped_data.reshape((num_samples, 20, num_hands * num_landmarks * 3))

# Convert class label to a numerical value (assuming a predefined class-to-index mapping)
class_mapping = {'0001': 0}  # Extend this mapping as needed
if class_label not in class_mapping:
    raise ValueError(f"Class label '{class_label}' not found in class mapping.")
class_label_numeric = class_mapping[class_label]

# Create an array of class labels, one for each sample
class_labels = np.full((num_samples,), class_label_numeric)

# Create an array of class names, one for each sample
class_names = np.full((num_samples,), class_label)

print(reshaped_data.shape)
print(reshaped_data)
print(class_labels.shape)
print(class_labels)
print(class_names.shape)
print(class_names)
# Output should be: (num_samples, 40, 252), (num_samples,), and (num_samples,)
