import os


def model_basename(model_name: str) -> str:
    """
    Extract the base name of the model from the full model name.
    """
    if "/" not in model_name:
        return model_name
    if model_name.count("/") > 1:
        return os.path.basename(model_name)
    return model_name.split("/")[1]
