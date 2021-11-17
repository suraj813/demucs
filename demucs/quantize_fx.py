from torch.quantization.quantize_fx import prepare_fx, convert_fx

import torch
from demucs.pretrained import load_pretrained

qdict = {
    'dynamic': {"": torch.quantization.default_dynamic_qconfig}, 
    'ptq':  {"": torch.quantization.get_default_qconfig("fbgemm")},
}


def main():
    fp32 = load_pretrained("demucs_quantized")
    fp32.eval()

    for i in range(len(fp32.encoder)):
        fp32.encoder[i] = prepare_fx(fp32.encoder[i], qdict['ptq'])
    for i in range(len(fp32.decoder)):
        fp32.decoder[i] = prepare_fx(fp32.decoder[i], qdict['ptq'])
        
    fp32.lstm = prepare_fx(fp32.lstm, qdict['dynamic'])

    #calibrate
    for _ in range(10):
        with torch.no_grad():
            fp32(torch.rand(1,2,44100))
    
    convert_fx(fp32)
    

