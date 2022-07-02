model_name = "bigscience/bloom"
input_device = 0
debug = False

if debug:
    input_device = "cpu"
    model_name = "gpt2"
