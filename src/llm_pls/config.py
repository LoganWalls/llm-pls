import os

port = int(os.getenv("LLMPLS_PORT", 8081))
model_name = os.getenv("LLMPLS_MODEL", "gpt2")

model_device = os.getenv("LLMPLS_DEVICE", "auto")
if model_device.isnumeric():
    model_device = int(model_device)

input_device = 0 if model_device == "auto" else model_device
debug = bool(os.getenv("LLMPLS_DEBUG", False))

if debug:
    model_device = "cpu"
    input_device = "cpu"
    model_name = "gpt2"
