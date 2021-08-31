import time
import torch
import torch.profiler
from .pretrained import load_pretrained
from .separate import load_track
from .utils import apply_model, apply_model_vec


# set device
dev = 'cpu'

# prepare input data
def load_inp(in_file='test.mp3'):
    in_track = load_track(in_file, dev, 2, 44100)
    ref = in_track.mean(0)
    in_track = (in_track - ref.mean()) / ref.std()
    return in_track

# load model
model = load_pretrained('demucs_quantized')
model.to(dev)

# initialize profiler

WARMUP = 1
ACTIVE = 1
REPEAT = 1 


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
    in_track = torch.hstack([in_track]*3)
    profile_naive(in_track)
    profile_vec(in_track)
