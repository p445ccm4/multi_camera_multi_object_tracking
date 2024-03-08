import numpy as np
import onnx
import onnxruntime as ort
import cv2
import argparse
from torchvision import transforms

# Function to preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (128, 256 ))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_image = transform(rgb_image)
    numpy_image = tensor_image.numpy()
    numpy_image = np.expand_dims(numpy_image, axis=0)
    return numpy_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test onnx model difference from pytorch model.")
    parser.add_argument("model_path", help="Path to the pytorch model file.")
    parser.add_argument("image_path", help="Path to the jpg image file.")
    args = parser.parse_args()

    # Load the ONNX model
    model_path = args.model_path
    model = onnx.load(model_path)

    # Check the model for any inconsistencies
    onnx.checker.check_model(model)
    print("onnx checked")

    # Create an ONNX runtime session
    ort_session = ort.InferenceSession(model.SerializeToString())

    # Load and preprocess the input image
    image_path = args.image_path
    input_tensor = preprocess_image(image_path)

    # Prepare the input name and input data
    input_name = ort_session.get_inputs()[0].name
    input_data = {input_name: input_tensor}

    # Run inference on the input data
    output_data = ort_session.run(None, input_data)

    # Print the output data
    print("Output data:", output_data)