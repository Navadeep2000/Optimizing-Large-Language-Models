import torch
import torch.onnx as onnx
from super_image import EdsrModel

# Initialize the EDSR model
model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=2)

# Unwrap the model from torch.nn.DataParallel if necessary
if isinstance(model, torch.nn.DataParallel):
    model = model.module

# Convert the PyTorch model to ONNX
dummy_input = torch.randn(1, 3, 256, 256)  # Adjust input size as needed
onnx_path = "edsr_model2.onnx"
onnx.export(model, dummy_input, onnx_path, export_params=True)
