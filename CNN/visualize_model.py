import torch
from torchviz import make_dot
from chess_model_class_attention import ChessModelAttention

# Create model instance
model = ChessModelAttention()

# Create a dummy input
batch_size = 1
x = torch.randn(batch_size, 13, 8, 8)  # [batch_size, channels, height, width]

# Generate output and visualization
y = model(x)
dot = make_dot(y, params=dict(model.named_parameters()))

# Save the visualization
dot.render("chess_model_architecture", format="png", cleanup=True)
print("Visualization has been saved as 'chess_model_architecture.png'") 