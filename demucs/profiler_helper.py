import time
import torch
from utils import vectorized_apply
from audio import AudioFile

dev = 'cpu'

# prepare input data
def load_inp(in_file='test.mp3'):
    in_track = load_track(in_file, dev, 2, 44100)
    ref = in_track.mean(0)
    in_track = (in_track - ref.mean()) / ref.std()
    return in_track, ref


def load_track(track, device, audio_channels, samplerate):
    wav = AudioFile(track).read(
        streams=0,
        samplerate=samplerate,
        channels=audio_channels).to(device)
    return wav

def stopwatch_infer_once(mdl, inp):
    t0 = time.time()
    _ = vectorized_apply(mdl, inp)
    return time.time()-t0


def run_profiler(model, inp, filename):
    WARMUP = 1
    ACTIVE = 1
    REPEAT = 1
    with torch.profiler.profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_log/'+filename),
        record_shapes=True,
        with_stack=True,
        schedule=torch.profiler.schedule(wait=0, warmup=WARMUP, active=ACTIVE, repeat=REPEAT)
    ) as prof:
        for _ in range((WARMUP + ACTIVE) * REPEAT):
            sources = vectorized_apply(model, inp, 8)
            prof.step()
    return sources


def print_size_of_model(model, name="Model"):
    torch.save(model.state_dict(), "temp.p")
    print(name, ' Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def module_equivalence(m1, m2, device=dev, rtol=1e-03, atol=1e-06, num_tests=50, input_size=(1,2,44100)):
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


def module_latency(m, device=dev, input_size=(1,2,44100), num_tests=10, warmup=1):
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


def profile_suite():
    x = load_inp()
