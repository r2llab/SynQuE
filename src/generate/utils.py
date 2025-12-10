def get_model_name(model: str) -> str:
    model_name = model.split("/")[-1].lower()
    return model_name