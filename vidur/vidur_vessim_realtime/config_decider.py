# config_decider.py

# This function decides the next Vidur config based on battery SoC and solar usage vs. demand.
# It uses simple thresholds to adjust model, batch size, and QPS.

def decide_next_config(
    current_config,
    battery_soc,
    last_chunk_usage,
    last_chunk_solar
):
    """
    Example logic to produce new Vidur config based on
    battery SoC and solar usage vs. demand.
    """
    new_config = current_config.copy()

    # Extract fields
    model_name = new_config.get("model_name", "Llama2-7b")
    batch_size = new_config.get("batch_size", 16)
    qps = new_config.get("qps", 20)

    # Sample ordered list of models
    model_order = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Meta-Llama-3-8B",
        "codellama/CodeLlama-34b-Instruct-hf"
    ]

    if model_name not in model_order:
        raise ValueError(f"Unknown model_name '{model_name}'. Must be one of: {model_order}")

    i = model_order.index(model_name) if model_name in model_order else 0

    # Simple thresholds
    if battery_soc < 0.2 or (last_chunk_solar < last_chunk_usage):
        # Decrease load
        if i > 0:
            i -= 1  # smaller model
        else:
            # already smallest
            if batch_size > 8:
                batch_size = max(1, batch_size // 2)
            else:
                qps = max(5, qps - 5)
    elif battery_soc > 0.8 and (last_chunk_solar > last_chunk_usage):
        # Increase load
        if i < len(model_order) - 1:
            i += 1
        else:
            # already largest
            if batch_size < 128:
                batch_size = min(128, batch_size * 2)
            else:
                qps = min(50, qps + 5)

    new_config["model_name"] = model_order[i]
    new_config["batch_size"] = batch_size
    new_config["qps"] = qps
    return new_config