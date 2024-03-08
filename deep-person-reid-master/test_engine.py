import argparse
import os
from pathlib import Path
import onnx
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

parser = argparse.ArgumentParser(description="Test TensorRT model difference from ONNX model.")
parser.add_argument("onnx_path", help="Path to the ONNX model file.")
parser.add_argument("engine_path", help="Path to the TensorRT engine file.")
args = parser.parse_args()

onnx_path = args.onnx_path
engine_path = args.engine_path

print("-------------------------check model---------------------------------------\n")

try:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    graph_output = onnx.helper.printable_graph(onnx_model.graph)
    with open("graph_output.txt", mode="w") as fout:
        fout.write(graph_output)	
except:
    print("Something went wrong")
	
import onnxruntime

ort_session = onnxruntime.InferenceSession(onnx_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

input = np.zeros((1, 3, 256, 128), dtype=np.float32)
ort_inputs = {ort_session.get_inputs()[0].name: input}
ort_outs = ort_session.run(None, ort_inputs)



TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

trt_engine = load_engine(engine_path)

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    return h_input, d_input, h_output, d_output

def do_inference(engine, h_input, d_input, h_output, d_output):
    with engine.create_execution_context() as context:
        cuda.memcpy_htod(d_input, h_input)
        context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
        cuda.memcpy_dtoh(h_output, d_output)
        return h_output

h_input, d_input, h_output, d_output = allocate_buffers(trt_engine)
np.copyto(h_input, input.ravel())
trt_output = do_inference(trt_engine, h_input, d_input, h_output, d_output)
trt_output = trt_output.reshape(ort_outs[0].shape)
# print('onnxs_output.shape', ort_outs[0].shape)
# print('trt_output.shape', trt_output.shape)

# Compare ONNX and TensorRT results
np.testing.assert_allclose(ort_outs[0], trt_output, rtol=1e-02, atol=1e-04)

print("ONNX model has been tested with TensorRT, and the result looks good!")
