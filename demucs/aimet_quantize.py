import torch
from .pretrained import load_pretrained
from .profiler import load_inp


# convert to onnx
model = load_pretrained("demucs_quantized_scripted")
inp, _ = load_inp()
torch.onnx.export(model, inp.unsqueeze(0), "demucs.onnx", opset_version=13, verbose=True)