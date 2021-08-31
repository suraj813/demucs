from .profile import *
import torch.quantization
import subprocess

model = load_model(True, True)

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear, nn.Conv1d}, dtype=torch.qint8
)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(quantized_model)