MAX_NEW_TOKEN = 100
TEMPERATURE = 0.01

MODEL_TO_ARGS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "max_model_len": 25000,
        "gpu_memory_utilization": 0.95,
    },
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "max_model_len": 25000,
        "gpu_memory_utilization": 0.95,
    },
    "Qwen/Qwen3-32B": {
        "max_model_len": 25000,
        "gpu_memory_utilization": 0.95,
    },
    "google/gemma-3-27b-it": {
        "max_model_len": 25000,
        "gpu_memory_utilization": 0.95,
    },
}
