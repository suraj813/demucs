import time
import torch
import torch.profiler
from .pretrained import load_pretrained
from .separate import load_track
from .utils import apply_model, apply_model_vec

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
    return in_track


# load model
def load_model(s=False, s_t=False):
    model = load_pretrained('demucs_quantized')
    model.to(dev)
    return model


def profile_naive(in_track):
    with torch.profiler.profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_log/demucs'),
        record_shapes=True,
        with_stack=True,
        schedule=torch.profiler.schedule(wait=0, warmup=WARMUP, active=ACTIVE, repeat=REPEAT)
    ) as prof:
        for i in range((WARMUP + ACTIVE) * REPEAT):
            sources = apply_model(model, in_track, shifts=0, split=True,
                                overlap=0.25, progress=False)
            prof.step()


def profile_vec(in_track):
    with torch.profiler.profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_log/demucs_vec'),
        record_shapes=True,
        with_stack=True,
        schedule=torch.profiler.schedule(wait=0, warmup=WARMUP, active=ACTIVE, repeat=REPEAT)
    ) as prof:
        for i in range((WARMUP + ACTIVE) * REPEAT):
            sources = apply_model_vec(model, in_track, 8)
            prof.step()

def profile_generic(inp, model, fn):
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

def compare_inp_lengths():
    from matplotlib import pyplot as plt
    inp0 = load_inp()
    y_naive = []
    y_vec = []
    x = range(1,8)
    for i in x:
        inp = torch.hstack([inp0]*i)

        t0 = time.time()
        _ , e1 = apply_model(model, inp, shifts=0, split=True, overlap=0.25, progress=False)
        diff1 = time.time() - t0
        y_naive.append(diff1)

        t0 = time.time()
        _, e2 = apply_model_vec(model, inp, 8)
        diff2 = time.time() - t0
        y_vec.append(diff2)

        print(f"e1 {e1} | Naive {diff1} | e2 {e2} | Vec {diff2}")

    plt.plot(x, y_naive, label="Naive")
    plt.plot(x, y_vec, label="Vec")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    in_track = load_inp()
    in_track = torch.hstack([in_track]*9)
    m1 = load_model()
    m2 = load_model(True, False)
    m3 = load_model(True, True)

    o1 = profile_generic(in_track, m1, 'eager_bigbat')
    o2 = profile_generic(in_track, m2, 's_bigbat')
    o3 = profile_generic(in_track, m3, 's_t_bigbat')

    assert torch.equal(o1, o2)
    assert torch.equal(o2, o3)
