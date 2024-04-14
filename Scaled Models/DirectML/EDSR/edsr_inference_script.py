import torch
import onnxruntime as rt
from PIL import Image
from torchvision.transforms.functional import to_tensor
import numpy as np
import time

# Load the preprocessed input image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256))  # Assuming EDSR model input size is 256x256
    image = to_tensor(image).unsqueeze(0)
    return image


# Load the ONNX model with DirectML backend
def load_model(onnx_path):
    ort_session = rt.InferenceSession(onnx_path, providers=['DmlExecutionProvider'])
    return ort_session


# Perform inference
def inference(model, inputs):
    start_time = time.perf_counter()
    ort_inputs = {model.get_inputs()[0].name: inputs.cpu().numpy()}
    ort_outputs = model.run(None, ort_inputs)
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    return ort_outputs, inference_time

# Save the output image
def save_image(output, output_path):
    output = output.squeeze(0)
    output = np.clip(output, 0, 1)  # Clip values to [0, 1] range
    output = (output * 255).astype(np.uint8)
    output_image = Image.fromarray(output.transpose(1, 2, 0))
    output_image.save(output_path)

# Main function
def main():
    # Input image path
    image_path = "input_image.jpeg"

    # Load the preprocessed input image
    inputs = load_image(image_path)

    # Load the ONNX model with DirectML backend
    load_start = time.perf_counter()
    onnx_path = "edsr_model.onnx"
    model = load_model(onnx_path)
    load_end = time.perf_counter()
    load_model_time = load_end - load_start

    # Perform inference
    outputs, inference_time = inference(model, inputs)

    # Save the output image
    output_path = "output_image.png"
    save_image(outputs[0], output_path)

    print("Output image saved successfully to:", output_path)
    print("Inference time:", inference_time, "seconds")
    print("Model Load Time:", load_model_time, "seconds")

if __name__ == "__main__":
    main()
