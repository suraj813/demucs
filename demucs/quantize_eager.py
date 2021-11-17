import torch
from torch import nn
from collections import OrderedDict
from demucs.pretrained import load_pretrained

def fuse_encoder(enc):
    """Fuses conv and relu models in the encoder sequential module"""
    enc.eval()
    for mod in enc.modules():
        if type(mod) == torch.nn.Sequential:
            torch.quantization.fuse_modules(mod, ['0','1'], inplace=True)
    return enc

def insert_stubs(seq, dequant_ix, quant_ix):
    layers = list(seq._modules.values())
    names = list(seq._modules.keys())
    layers.insert(dequant_ix, torch.quantization.DeQuantStub())
    names.insert(dequant_ix, f'dequant{dequant_ix}')
    layers.insert(quant_ix, torch.quantization.QuantStub())
    names.insert(quant_ix, f'quant{quant_ix}')
    seq._modules = OrderedDict(zip(names, layers))
    return seq


def calibrate(module, calibration_input):
    """Calibrate the model by passing a `calibration_input` for the observer modules.
    Convert submodules to the quantized version."""
    x = calibration_input
    with torch.inference_mode():
        try:
            module(x)
        except NotImplementedError:
            for mod in module:
                x = mod(x)

def main():
    # fuse
    fp32 = load_pretrained("demucs_quantized")
    fuse_encoder(fp32.encoder)

    # prepare
    qconfig = torch.quantization.default_qconfig
    fp32.encoder.qconfig = qconfig
    fp32.decoder.qconfig = qconfig

    for seq in fp32.encoder:
        seq = insert_stubs(seq, 3, 0)
    torch.quantization.prepare(fp32.encoder, inplace=True)

    for seq in fp32.decoder:
        seq = insert_stubs(seq, 4, 2)
        seq = insert_stubs(seq, 1, 0)
    torch.quantization.prepare(fp32.decoder, inplace=True)

    # calibrate
    for _ in range(10):
        x = torch.rand(1,2,44100) 
    calibrate(fp32.encoder, x)

    for _ in range(10):
        x = torch.rand(1,2048,9) 
    calibrate(fp32.decoder, x)

    # convert
    torch.quantization.convert(fp32.encoder, mapping=torch.quantization.get_default_static_quant_module_mappings())
    torch.quantization.convert(fp32.decoder, mapping=torch.quantization.get_default_static_quant_module_mappings())

    # lstm
    torch.quantization.quantize_dynamic(fp32, {nn.LSTM, nn.Linear}, inplace=True)

    return fp32

