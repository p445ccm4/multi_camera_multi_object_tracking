import torchreid
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="turn Pytorch model to ONNX model.")
parser.add_argument("model_path", help="Path to the pytorch model file.")
args = parser.parse_args()

file_name = os.path.basename(args.model_path).split('.')[0]


torchreid.models.show_avai_models()

model = torchreid.models.build_model(name= file_name, num_classes=1000)

torchreid.utils.load_pretrained_weights(model, args.model_path)

from torch.autograd import Variable
import torch
import onnx

save_onnx_path = Path(args.model_path).with_suffix('.onnx')

input_name = ['input']
output_name = ['output']
input = Variable(torch.randn(1, 3, 256, 128))
torch.onnx.export(model, input, save_onnx_path, input_names=input_name,output_names=output_name, verbose=True, export_params=True)

# The model after convert only 10633KBytes, while the pytorch model got 16888KBytes

onnx_model = onnx.load(save_onnx_path)
onnx.checker.check_model(onnx_model)
print('onnx model generate')

