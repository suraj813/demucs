import torch
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from model_fx import ResampleNonTraceable, load_model
from demucs.profiler_helper import module_latency, print_size_of_model, module_equivalence

def calibrate_prepared_model(prepared_model):
    for _ in range(20):
        n = torch.randint(1, 10, (1,)).item()
        x = torch.rand(1,2,44100*n)
        _ = prepared_model(x)

fp_model = load_model()
fp_model.eval()

qconfig = get_default_qconfig("fbgemm")

# object_type specifies modules that can't be quantized
qconfig_dict = {
    "": qconfig,  # global config
    "object_type": [(torch.nn.ConvTranspose1d, None), (torch.nn.GLU, None)]  # tuple[1]=None means no qconfig for this module
}

# "non_traceable_module_class" specify modules that can't be symbolically traced
prepare_custom_config_dict = {
    "non_traceable_module_class": [ResampleNonTraceable]  
}

prepared_model = prepare_fx(fp_model, qconfig_dict, prepare_custom_config_dict=prepare_custom_config_dict)
prepared_model.eval()
calibrate_prepared_model(prepared_model)


"""NotImplementedError: Could not run 'aten::empty.memory_format' with arguments from the 'QuantizedCPU' backend. 
This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build)."""
quantized_model = convert_fx(prepared_model)
calibrate_prepared_model(quantized_model)

assert module_equivalence(quantized_model, fp_model, num_tests=10) 
print_size_of_model(fp_model, "FP32")
print("Latency: ", module_latency(fp_model))
print()
print_size_of_model(quantized_model, "FX Quantized")
print("Latency: ", module_latency(quantized_model))


