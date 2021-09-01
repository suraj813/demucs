import time
import torch
from torch import nn
import torch.profiler
from .pretrained import load_pretrained
from .separate import load_track
from .utils import apply_model, apply_model_vec
from .quantize import QuantizedDemucs
import subprocess, os, copy

# set device
dev = 'cpu'

WARMUP = 1
ACTIVE = 1
REPEAT = 1 

# prepare input data
def load_inp(in_file='test.mp3'):
    in_track = load_track(in_file, dev, 2, 44100)
    ref = in_track.mean(0)
    in_track = (in_track - ref.mean()) / ref.std()
    return in_track, ref


# load model
def load_model(s=False, t=False):
    name = "demucs_quantized"
    if s:
        name += "_scripted"
    if t:
        name += "_traced"
    model = load_pretrained(name)
    model.to(dev)
    return model


def profile(inp, model, fn):
    with torch.profiler.profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_log/'+fn),
        record_shapes=True,
        with_stack=True,
        schedule=torch.profiler.schedule(wait=0, warmup=WARMUP, active=ACTIVE, repeat=REPEAT)
    ) as prof:
        for i in range((WARMUP + ACTIVE) * REPEAT):
            sources = apply_model_vec(model, inp, 8)
            prof.step()
    return sources


def encode(sources, ref, folder='.', fmt='ogg'):
    def postprocess(inference_output):
        stems = []
        for source in inference_output:
            source = source / max(1.01 * source.abs().max(), 1)  # source.max(dim=1).values.max(dim=-1)
            source = (source * 2**15).clamp_(-2**15, 2**15 - 1).short()
            source = source.cpu().numpy()
            stems.append(source)
        return stems
    sources = sources * ref.std() + ref.mean()
    sources = postprocess(sources)
    cmd = f"sox --multi-threaded -t s16 -r 44100 -c 2 - -t {fmt} -r 44100 -b 16 -c 2 -"
    cmd_a = cmd.split(' ')
    source_names = ["drums", "bass", "other", "vocals"]
    for i,name in enumerate(source_names):
        array = sources[i]
        key = folder + '/' + name + '.' + fmt
        handle = subprocess.Popen(cmd_a, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = handle.communicate(array.tobytes(order='F'))
        with open(key, 'wb') as f:
            f.write(out)
    return err


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def module_equivalence(m1, m2, device='cpu', rtol=1e-03, atol=1e-06, num_tests=50, input_size=(1,2,44100)):
    m1 = m1.to(device)
    m2 = m2.to(device)
    m1.eval()
    m2.eval()
    failed = 0
    for _ in range(num_tests):
        x = torch.rand(input_size)
        with torch.inference_mode():
            y1 = m1(x)
            y2 = m2(x)
        if not torch.allclose(y1, y2, rtol=rtol, atol=atol):
            failed += 1
    if failed > 0:
        print(f"Failed: {failed}/{num_tests}")
        return False
    return True


def module_latency(m, device='cpu', input_size=(1,2,44100), num_tests=10, warmup=1):
    m = m.to(device)
    m.eval()
    x = torch.rand(input_size)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = m(x)

    t0 = time.time()
    with torch.inference_mode():
        for _ in range(num_tests):
            _ = m(x)
    elapsed = time.time() - t0

    return elapsed/num_tests


def profile_eager_jit():
    in_track, ref = load_inp()
    in_track = torch.hstack([in_track]*3)
    m1 = load_model()
    m2 = load_model(True, False)
    m3 = load_model(True, True)

    o1 = profile(in_track, m1, 'eager')
    encode(o1, ref, 'eager', 'mp3')
    o2 = profile(in_track, m2, 'scripted-only')
    encode(o2, ref, 'sc', 'mp3')
    o3 = profile(in_track, m3, 'scripted-traced')
    encode(o3, ref, 'sc_tr', 'mp3')

    assert torch.equal(o1, o2)
    assert torch.equal(o2, o3)


def profile_dyn_quantized():
    in_track, ref = load_inp()
    in_track = torch.hstack([in_track]*3)
    
    m = load_model()
    qm = torch.quantization.quantize_dynamic(
        m, {nn.LSTM, nn.Linear, nn.Conv1d}, dtype=torch.qint8
    )

    print_size_of_model(m)
    print_size_of_model(qm)

    o4 = profile(in_track, qm, 'd_quant')
    encode(o4, ref, 'd_quant', 'mp3')

def get_fused():
    m = load_model()        # Load pretrained model
    fused_m = copy.deepcopy(m)
    
    m.eval()                # Switch to eval
    fused_m.eval()

    # Layer fusion -- currently we can only fuse conv+relu in the encoder
    for mod in fused_m.encoder.modules():
        if type(mod) == torch.nn.Sequential:
            torch.quantization.fuse_modules(mod, ['0', '1'], inplace=True)

    # Verify if encoder layer fusion is correct
    # print(fused_m.encoder)
    assert module_equivalence(
        nn.Sequential(*fused_m.encoder), nn.Sequential(*m.encoder)
    )
    return m, fused_m


def get_quant(fused_m):
    # Apply stubs to the input and output. \
    # This collects quantization statistics for inputs and outputs. \
    # By default, pytorch only does this for weights (and activations?)
    qm = QuantizedDemucs(fused_m)

     # Init quant config
    # qm.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    qm.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(qm, inplace=True)

    # calibrate
    inp, ref = load_inp()  # pass representative data!
    src = apply_model_vec(qm, inp) 
    encode(src, ref, 'quant', 'mp3')

    # Convert calibrated fp32 to qint model
    torch.quantization.convert(qm, inplace=True)
    print_size_of_model(qm)
    return qm


if __name__ == "__main__":
    m, fused = get_fused()
    qm = get_quant(fused)
    print(qm)
