port = 8081
model_name = "bigscience/bloom"
model_device = "auto"
input_device = 0 if model_device == "auto" else model_device
debug = True

if debug:
    model_device = "cpu"
    input_device = "cpu"
    model_name = "gpt2"
