from locale import normalize
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import torchvision.transforms as transforms
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def preprocess_image(image, target_shape=(256, 128)):
    resized_image = cv2.resize(image.astype(np.float32)/255., target_shape)
    norm = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    normalized_image = norm(resized_image).unsqueeze(0).float()
    # print(resized_image.shape)#same
    # means = [0.485, 0.456, 0.406]
    # stds = [0.229, 0.224, 0.225]
    # normalized_image = (resized_image - means) / stds
    # print(normalized_image)
    return normalized_image

def load_engine(engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    print(1)
    return engine

def get_binding_shape(engine, binding_index):
    if hasattr(engine, "get_tensor_shape"):
        tensor_name = engine.get_tensor_name(binding_index)
        return engine.get_tensor_shape(tensor_name)
    elif binding_index < engine.num_inputs:
        return engine.get_input_shape(binding_index)
    else:
        return engine.get_output_shape(binding_index - engine.num_inputs)

def infer(engine, input_image):
    cuda.init()

    context = engine.create_execution_context()

    input_shape = list(get_binding_shape(engine, 0))
    output_shape = list(get_binding_shape(engine, 1))

    # Replace dynamic batch size with actual batch size
    input_shape[0] = 1
    output_shape[0] = 1

    # print("Input shape:", input_shape)
    # print("Output shape:", output_shape)

    h_input = input_image.view(input_shape).contiguous().type(torch.float32)
    h_output = torch.empty(output_shape, dtype=torch.float32)

    input_tensor_name = engine.get_tensor_name(0)
    if hasattr(context, "set_input_shape"):
        context.set_input_shape(input_tensor_name, tuple(input_shape))  # Set input binding dimensions using set_input_shape
    else:
        context.set_binding_shape(0, tuple(input_shape))  # Fallback to set_binding_shape for older TensorRT versions

    # Create a GPU tensor and copy the data from the host tensor
    d_input_torch = torch.cuda.FloatTensor(*input_shape).copy_(h_input)
    # print(d_input_torch)#same

    # Allocate an empty GPU tensor for the output
    d_output_torch = torch.cuda.FloatTensor(*output_shape)

    context.execute_v2(bindings=[d_input_torch.data_ptr(), d_output_torch.data_ptr()])  # Pass GPU tensors directly to execute_v2

    # Copy the data from the GPU output tensor back to the host
    h_output.copy_(d_output_torch)

    return h_output.numpy()  # Convert the output tensor back to a numpy array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test tensorrt model output on image.")
    parser.add_argument("engine_path", help="Path to the tensorrt model file.")
    parser.add_argument("image_path", help="Path to the jpg image file.")
    args = parser.parse_args()

    # Load TensorRT engine
    engine_path = args.engine_path
    engine = load_engine(engine_path)

    # Read and preprocess input image
    image_path = args.image_path
    input_image_rgb = cv2.imread(image_path)[:,:,(2,1,0)]
    # print(input_image_rgb) same
    # input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    preprocessed_image = preprocess_image(input_image_rgb)
    # print(preprocessed_image)#same

    # Add batch dimension and transpose to CHW format
    # input_batch = np.expand_dims(, axis=0)

    # Run inference
    output = infer(engine, preprocessed_image)
    print('Output:', output)


