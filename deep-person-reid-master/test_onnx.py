import torchreid
import sys
import torch
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="test onnx model difference from pytorch model.")
parser.add_argument("model_path", help="Path to the pytorch model file.")
args = parser.parse_args()

file_name = os.path.basename(args.model_path).split('.')[0]

torchreid.models.show_avai_models()

model = torchreid.models.build_model(name=file_name, num_classes=1041)
torchreid.utils.load_pretrained_weights(model, args.model_path)
idx = 0
# for m, p in model.named_parameters():
#     print(m, torch.isnan(p).any())
#     idx += 1

print("total layers: ", idx)

model.eval()

from torch.autograd import Variable
import torch
import onnx

# An example input you would normally provide to your model's forward() method.
input = torch.zeros(1, 3, 256, 128)
raw_output = model(input)

onnx_path = Path(args.model_path).with_suffix('.onnx')
torch.onnx.export(model, input, onnx_path, verbose=False, export_params=True)

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
import numpy as np

ort_session = onnxruntime.InferenceSession(onnx_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
ort_outs = ort_session.run(None, ort_inputs)	

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(raw_output), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
