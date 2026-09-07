# Variables from your model
batch_size = 16  # From your code
seq_len = 20  # Sequence length (frames)
hidden_size = 512  # Hidden size
num_layers = 2  # Number of transformer layers
num_classes = 190  # Number of classes

# 1. Self-attention computation per transformer layer:
# Complexity per layer: O(N^2 * d)
attention_flops_per_layer = 2 * (seq_len ** 2) * hidden_size  # O(N^2 * d)

# 2. Feedforward network computation per transformer layer:
# Complexity per layer: O(2 * d^2)
feedforward_flops_per_layer = 4 * (hidden_size ** 2)  # O(2 * d^2)

# Total computation per transformer layer (both attention and feedforward):
flops_per_layer = attention_flops_per_layer + feedforward_flops_per_layer

# 3. Total computation for all layers:
total_flops_layers = num_layers * flops_per_layer

# 4. Final classification layer:
# Complexity: O(d * num_classes)
classification_flops = hidden_size * num_classes

# 5. Total GFLOPs (converting to GFLOPs = 10^-9)
total_flops = (total_flops_layers + classification_flops) * batch_size
gflops = total_flops / 1e9  # Convert to GFLOPs

print(f"Estimated GFLOPs per forward pass (2layers) for KARSL190: {gflops:.4f} GFLOPs")

# Variables from your model
batch_size = 16  # From your code
seq_len = 20  # Sequence length (frames)
hidden_size = 256  # Hidden size
num_classes = 190  # Number of classes

# 1. Self-attention computation per transformer layer:
# Complexity per layer: O(N^2 * d)
attention_flops_per_layer = 2 * (seq_len ** 2) * hidden_size  # O(N^2 * d)

# 2. Feedforward network computation per transformer layer:
# Complexity per layer: O(2 * d^2)
feedforward_flops_per_layer = 4 * (hidden_size ** 2)  # O(2 * d^2)

# Total computation per transformer layer (both attention and feedforward):
flops_per_layer = attention_flops_per_layer + feedforward_flops_per_layer

# 3. Total computation for all layers:
total_flops_layers = num_layers * flops_per_layer

# 4. Final classification layer:
# Complexity: O(d * num_classes)
classification_flops = hidden_size * num_classes

# 5. Total GFLOPs (converting to GFLOPs = 10^-9)
total_flops = (total_flops_layers + classification_flops) * batch_size
gflops = total_flops / 1e9  # Convert to GFLOPs

print(f"Estimated GFLOPs per forward pass (1layer) for KARSL190: {gflops:.4f} GFLOPs")


# Variables from your model
batch_size = 16  # From your code
seq_len = 40  # Sequence length (frames)
hidden_size = 512  # Hidden size
num_layers = 2  # Number of transformer layers
num_classes = 226  # Number of classes

# 1. Self-attention computation per transformer layer:
# Complexity per layer: O(N^2 * d)
attention_flops_per_layer = 2 * (seq_len ** 2) * hidden_size  # O(N^2 * d)

# 2. Feedforward network computation per transformer layer:
# Complexity per layer: O(2 * d^2)
feedforward_flops_per_layer = 4 * (hidden_size ** 2)  # O(2 * d^2)

# Total computation per transformer layer (both attention and feedforward):
flops_per_layer = attention_flops_per_layer + feedforward_flops_per_layer

# 3. Total computation for all layers:
total_flops_layers = num_layers * flops_per_layer

# 4. Final classification layer:
# Complexity: O(d * num_classes)
classification_flops = hidden_size * num_classes

# 5. Total GFLOPs (converting to GFLOPs = 10^-9)
total_flops = (total_flops_layers + classification_flops) * batch_size
gflops = total_flops / 1e9  # Convert to GFLOPs

print(f"Estimated GFLOPs per forward pass (2layers) for AUTSL: {gflops:.4f} GFLOPs")
