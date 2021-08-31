import time
import torch
import torch.profiler
from .pretrained import load_pretrained
from .separate import load_track
from .utils import apply_model, apply_model_vec
import subprocess

# set device
dev = 'cpu'

WARMUP = 1
ACTIVE = 1
REPEAT = 1 

# prepare input data
def load_inp(in_file='original.mp3'):
    in_track = load_track(in_file, dev, 2, 48000)
    ref = in_track.mean(0)
    in_track = (in_track - ref.mean()) / ref.std()
    return in_track


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


def postprocess(sources, folder='.', fmt='ogg'):
    cmd = f"sox --multi-threaded -t s16 -r 44100 -c 2 - -t {fmt} -r 44100 -b 16 -c 2 -"
    cmd_a = cmd.split(' ')
    source_names = ["drums", "bass", "other", "vocals"]
    for i,name in enumerate(source_names):
        array = sources[i].numpy().T
        key = folder + '/' + name + '.' + fmt
        handle = subprocess.Popen(cmd_a, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = handle.communicate(array.tobytes(order='F'))
        with open(key, 'wb') as f:
            f.write(out)
    return err

if __name__ == "__main__":
    in_track = load_inp()
    # in_track = torch.hstack([in_track]*3)
    m1 = load_model()
    m2 = load_model(True, False)
    m3 = load_model(True, True)

    o1 = profile(in_track, m1, 'eager')
    postprocess(o1, 'eager', 'mp3')
    o2 = profile(in_track, m2, 'scripted-only')
    postprocess(o2, 'sc', 'mp3')
    o3 = profile(in_track, m3, 'scripted-traced')
    postprocess(o3, 'sc_tr', 'mp3')

    assert torch.equal(o1, o2)
    assert torch.equal(o2, o3)
