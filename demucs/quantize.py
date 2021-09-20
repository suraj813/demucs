from .profiler import *
from collections import OrderedDict

class QuantizedEncoder(nn.Module):
    def __init__(self, fused_encoder_fp32):
        super(QuantizedEncoder, self).__init__()
        self.fused = copy.deepcopy(fused_encoder_fp32)
        for seq in self.fused:
            od = seq._modules
            layers = list(od.values())
            layers.insert(0, torch.quantization.QuantStub())
            layers.insert(4, torch.quantization.DeQuantStub())
            od2 = OrderedDict(zip([str(x) for x in range(len(layers))], layers))
            seq._modules = od2
        
    def forward(self, x):
        for mod in self.fused:
            x = mod(x)
        return x

class QuantizedDecoder(nn.Module):
    def __init__(self, decoder_fp32):
        super(QuantizedDecoder, self).__init__()
        self.fused = copy.deepcopy(decoder_fp32)
        for seq in self.fused:
            od = seq._modules
            layers = list(od.values())
            layers.insert(0, torch.quantization.QuantStub())
            layers.insert(2, torch.quantization.DeQuantStub())
            layers.insert(4, torch.quantization.QuantStub())
            layers.insert(6, torch.quantization.DeQuantStub())
            od2 = OrderedDict(zip([str(x) for x in range(len(layers))], layers))
            seq._modules = od2
        
    def forward(self, x):
        for mod in self.fused:
            x = mod(x)
        return x


def fuse_encoder(enc):
    enc.eval()
    for mod in enc.modules():
        if type(mod) == torch.nn.Sequential:
            torch.quantization.fuse_modules(mod, ['0','1'], inplace=True)
    return enc


def prep_encoder(fused_enc):
    qe = QuantizedEncoder(fused_enc)
    qe.qconfig = torch.quantization.default_qconfig
    qe = torch.quantization.prepare(qe)
    return qe.fused

def prep_decoder(decoder):
    qd = QuantizedDecoder(decoder)
    qd.qconfig = torch.quantization.default_qconfig
    qd = torch.quantization.prepare(qd)
    return qd.fused

def convert_quant(qmodel, calibrate_inp=None):
    if calibrate_inp is not None:
        _ = apply_model_vec(qmodel, calibrate_inp, 8) 
    torch.quantization.convert(qmodel.encoder, inplace=True)
    torch.quantization.convert(qmodel.decoder, inplace=True)

def prep_quant_pipeline(model, dyn=False, decoder=False, calibrate_inp=None):
    fused_enc = fuse_encoder(model.encoder)
    model.encoder = prep_encoder(fused_enc)
    if decoder:
        model.decoder = prep_decoder(model.decoder)
    convert_quant(model, calibrate_inp)
    if dyn:
        torch.quantization.quantize_dynamic(
            model, {nn.LSTM, nn.Linear, nn.Conv1d}, dtype=torch.qint8, inplace=True
        )
    return model

def load_quantized_model_from_disk(pkl, dyn=False, decoder=False):
    model = Demucs([1,1,1,1])
    model = prep_quant_pipeline(model, dyn, decoder)
    model.load_state_dict(torch.load(pkl))
    return model


def main():
    MODEL = load_demucs_model()
    inp, ref = load_inp('original.ogg')

    dynq =  torch.quantization.quantize_dynamic(
            MODEL, {nn.LSTM, nn.Linear, nn.Conv1d}, dtype=torch.qint8
        )

    qencoder = copy.deepcopy(MODEL)
    fused_enc = fuse_encoder(qencoder.encoder)
    qencoder.encoder = prep_encoder(fused_enc)
    convert_quant(qencoder, inp)
    dyn_qencoder = torch.quantization.quantize_dynamic(
            qencoder, {nn.LSTM, nn.Linear, nn.Conv1d}, dtype=torch.qint8
        ) # dynamic mapping doesn't have conv1d yet there is a minor speedup by including it

    q_encdec = copy.deepcopy(MODEL)
    fused_enc = fuse_encoder(q_encdec.encoder)
    q_encdec.encoder = prep_encoder(fused_enc)
    q_encdec.decoder = prep_decoder(q_encdec.decoder)
    convert_quant(q_encdec, inp)
    dyn_q_encdec = torch.quantization.quantize_dynamic(
            q_encdec, {nn.LSTM, nn.Linear, nn.Conv1d}, dtype=torch.qint8
        ) # dynamic mapping doesn't have conv1d yet there is a minor speedup by including it


    names = ['MODEL', 'dynq', 'qencoder', 'dyn_qencoder', 'q_encdec', 'dyn_q_encdec']
    models = [MODEL, dynq, qencoder, dyn_qencoder, q_encdec, dyn_q_encdec]
    for name, mod in zip(names,models):
        print(f"Model: {name}")
        print(f"Size: {print_size_of_model(mod)}")
        print(f"Latency on 1s sample: {module_latency(mod)}")
        t0 = time.time()
        sources = apply_model_vec(mod, inp, 8)
        elapsed = time.time() - t0
        print(f"Latency on 7m song: {elapsed}")
        encode(sources, ref, folder=f'quantized_outputs/{name}')
        print(f"Audio output folder: quantized_outputs/{name}")
        torch.save(mod.state_dict(), f'quantized_outputs/{name}.pt')
        print(f"Model saved to: quantized_outputs/{name}.pt")



# Model: MODEL
# Size (MB): 1062.738911
# Latency on 1s sample: 0.6736747026443481
# Latency on 7m song: 221.8002438545227
# Audio output folder: quantized_outputs/MODEL
# Model saved to: quantized_outputs/MODEL.pt

# Model: dynq
# Size (MB): 534.258841
# Latency on 1s sample: 0.3377014875411987
# Latency on 7m song: 153.23943495750427
# Audio output folder: quantized_outputs/dynq
# Model saved to: quantized_outputs/dynq.pt

# Model: qencoder
# Size (MB): 962.157493
# Latency on 1s sample: 0.6052535057067872
# Latency on 7m song: 191.35514116287231
# Audio output folder: quantized_outputs/qencoder
# Model saved to: quantized_outputs/qencoder.pt

# Model: dyn_qencoder
# Size (MB): 433.677233
# Latency on 1s sample: 0.2500915050506592
# Latency on 7m song: 113.88189005851746
# Audio output folder: quantized_outputs/dyn_qencoder
# Model saved to: quantized_outputs/dyn_qencoder.pt

# Model: q_encdec
# Size (MB): 794.478037
# Latency on 1s sample: 0.46161298751831054
# Latency on 7m song: 148.61190700531006
# Audio output folder: quantized_outputs/q_encdec
# Model saved to: quantized_outputs/q_encdec.pt

# Model: dyn_q_encdec
# Size (MB): 265.997777
# Size: None
# Latency on 1s sample: 0.12788548469543456
# Latency on 7m song: 77.12007713317871
# Audio output folder: quantized_outputs/dyn_q_encdec
# Model saved to: quantized_outputs/dyn_q_encdec.pt