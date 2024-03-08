import sys
import argparse
import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import os
import time
from PIL import Image
import cv2
import torchvision
from pathlib import Path



def get_img_np_nchw(filename):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (256, 128))
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    #normalize image to [0,1]
    img_np = np.array(image_cv, dtype=float) / 255.

    
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])
    img_np_nchw = np.expand_dims(img_np_t, axis=0)
    return img_np_nchw


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(onnx_file_path="", engine_file_path="", \
               fp16_mode=False, int8_mode=False, save_engine=False,
               ):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(explicit_batch) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser,\
                builder.create_builder_config() as builder_config:

            runtime = trt.Runtime(TRT_LOGGER)

            
            # pdb.set_trace()
            if fp16_mode:
                builder_config.set_flag(trt.BuilderFlag.FP16)

            if int8_mode:
                # To be updated
                raise NotImplementedError

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')

            # 根据 osnet，reshape 输入数据的形状
            # network.get_input(0).shape = [1, 3, 256, 128] #new add
            
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            
            # last_layer = network.get_layer(network.num_layers - 1)
            # network.mark_output(last_layer.get_output(0))

            # 序列化模型
            serialized_engine = builder.build_serialized_network(network, builder_config) #new add
            
            # 反序列化
            engine = runtime.deserialize_cuda_engine(serialized_engine) #new add
            # engine = builder.build_engine(network, builder_config)

            print("Completed creating Engine")

            # 写入文件
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(serialized_engine)
                print(f'saved to {engine_file_path}')
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            # print(runtime.deserialize_cuda_engine(f.read())) error
            trt.init_libnvinfer_plugins(TRT_LOGGER, '') 
            # 反序列化
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(save_engine)


def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create tensorrt engine using onnx model.")
    parser.add_argument("model_path", help="Path to the onnx model file.")
    parser.add_argument("image_path", help="Path to the jpg image file.")
    args = parser.parse_args()
    onnx_model_path = args.model_path
    image_path = args.image_path

    TRT_LOGGER = trt.Logger()  # This logger is required to build an engine 

    img_np_nchw = get_img_np_nchw(image_path)
    img_np_nchw = img_np_nchw.astype(dtype=np.float32)

    # These two modes are dependent on hardwares
    fp16_mode = True
    int8_mode = False

    trt_engine_path = Path(args.model_path).with_suffix('.engine')

    # Build an engine
    engine = get_engine(onnx_model_path, trt_engine_path, fp16_mode, int8_mode, save_engine=True)
    # Create the context for this engine
    context = engine.create_execution_context()
    # Allocate buffers for input and output
    inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host # bindings

    # Do inference
    shape_of_output = (1, 512)

    # Load data to the buffer
    inputs[0].host = img_np_nchw.reshape(-1)

    # inputs[1].host = ... for multiple input
    t1 = time.time()
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)  # numpy data
    t2 = time.time()
    
    print("Inference time with the TensorRT engine: {}".format(t2 - t1))
    feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)
    print("output: ", feat)
    print('TensorRT ok')


