import time
import torch
import torch.profiler
from .pretrained import load_pretrained
from .separate import load_track
from .utils import apply_model #, apply_model_vec
import subprocess

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


def compare():
    in_track = load_inp()
    # in_track = torch.hstack([in_track]*3)
    m1 = load_model()
    m2 = load_model(True, False)
    m3 = load_model(True, True)

    o1 = profile(in_track, m1, 'eager')
    encode(o1, 'eager', 'mp3')
    o2 = profile(in_track, m2, 'scripted-only')
    encode(o2, 'sc', 'mp3')
    o3 = profile(in_track, m3, 'scripted-traced')
    encode(o3, 'sc_tr', 'mp3')

    assert torch.equal(o1, o2)
    assert torch.equal(o2, o3)


if __name__ == "__main__":
    compare()
