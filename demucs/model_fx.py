import math
import julius
import torch
from torch import nn
import io, zlib
from utils import capture_init, center_trim

from torch.fx import wrap
wrap('center_trim')

def rescale_module(module, reference):
    def rescale_conv(conv, reference):
        std = conv.weight.std().detach()
        scale = (std / reference)**0.5
        conv.weight.data /= scale
        if conv.bias is not None:
            conv.bias.data /= scale

    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class BLSTM(nn.Module):
    def __init__(self, dim, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x


class ResampleNonTraceable(nn.Module):
    """Module to wrap (non-traceable) third-party pythonlib calls"""
    def forward(self, x, d1: int, d2: int, active: bool):  
        if active:
            return julius.resample_frac(x, d1, d2)
        return x


class Demucs(nn.Module):
    @capture_init
    def __init__(self,
                 sources,
                 audio_channels=2,
                 channels=64,
                 depth=6,
                 rewrite=True,
                 glu=True,
                 rescale=0.1,
                 resample=True,
                 kernel_size=8,
                 stride=4,
                 growth=2.,
                 lstm_layers=2,
                 context=3,
                 normalize=False,
                 samplerate=44100,
                 segment_length=4 * 10 * 44100):
        """
        Args:
            sources (list[str]): list of source names
            audio_channels (int): stereo or mono
            channels (int): first convolution channels
            depth (int): number of encoder/decoder layers
            rewrite (bool): add 1x1 convolution to each encoder layer
                and a convolution to each decoder layer.
                For the decoder layer, `context` gives the kernel size.
            glu (bool): use glu instead of ReLU
            resample_input (bool): upsample x2 the input and downsample /2 the output.
            rescale (int): rescale initial weights of convolutions
                to get their standard deviation closer to `rescale`
            kernel_size (int): kernel size for convolutions
            stride (int): stride for convolutions
            growth (float): multiply (resp divide) number of channels by that
                for each layer of the encoder (resp decoder)
            lstm_layers (int): number of lstm layers, 0 = no lstm
            context (int): kernel size of the convolution in the
                decoder before the transposed convolution. If > 1,
                will provide some context from neighboring time
                steps.
            samplerate (int): stored as meta information for easing
                future evaluations of the model.
            segment_length (int): stored as meta information for easing
                future evaluations of the model. Length of the segments on which
                the model was trained.
        """

        super().__init__()
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.resample = resample
        self.channels = channels
        self.normalize = normalize
        self.samplerate = samplerate
        self.segment_length = segment_length
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1

        in_channels = audio_channels
        for index in range(depth):
            encode = []
            encode += [nn.Conv1d(in_channels, channels, kernel_size, stride), nn.ReLU()]
            if rewrite:
                encode += [nn.Conv1d(channels, ch_scale * channels, 1), activation]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = len(self.sources) * audio_channels
            if rewrite:
                decode += [nn.Conv1d(channels, ch_scale * channels, context), activation]
            decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)
        channels = in_channels

        self.lstm = BLSTM(channels, lstm_layers)

        if rescale:
            rescale_module(self, reference=rescale)

        self.non_traceable_resample = ResampleNonTraceable()


    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length when context = 1. If context > 1,
        the two signals can be center trimmed to match.

        For training, extracts should have a valid length.For evaluation
        on full tracks we recommend passing `pad = True` to :method:`forward`.
        """
        if self.resample:
            length *= 2
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        if self.resample:
            length = math.ceil(length / 2)
        return int(length)


    def forward(self, x):
        mean = 0
        std = 1
        x = (x - mean) / (1e-5 + std)
        x = self.non_traceable_resample(x, 1, 2, self.resample)

        # Encoder Pass
        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)
        
        # LSTM Pass
        x = self.lstm(x)

        # Decoder Pass
        for decode in self.decoder:
            skip = center_trim(saved.pop(-1), x) # Align skip-connection size with decoder input
            x = x + skip
            x = decode(x)
        
        x = self.non_traceable_resample(x, 2, 1, self.resample)
        x = x * std + mean
        x = x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))
        return x


def load_model(is_quantized=False):
    model = Demucs([1,1,1,1])
    if is_quantized:
        from diffq import DiffQuantizer
        model_weights_url = "https://dl.fbaipublicfiles.com/demucs/v3.0/demucs_quantized-07afea75.th"
    else:
        model_weights_url = "https://dl.fbaipublicfiles.com/demucs/v3.0/demucs-e07c671f.th"

    state = torch.hub.load_state_dict_from_url(model_weights_url, map_location='cpu', check_hash=True)

    if is_quantized:
        quantizer = DiffQuantizer(model, group_size=8, min_size=1)
        buf = io.BytesIO(zlib.decompress(state["compressed"]))
        state = torch.load(buf, "cpu")
        quantizer.restore_quantized_state(state)
        quantizer.detach() 
    else:
        model.load_state_dict(state)
    model.eval()
    return model