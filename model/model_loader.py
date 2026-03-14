import pickle
import json
from huggingface_hub import hf_hub_download

def load_artifacts():
    model_path = hf_hub_download(
        repo_id="sidhantmanitiwari/car-price-rfr",
        filename="random_forest_regressor.pkl"
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open("model_label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open("car_models.json", "r") as f:
        car_models = json.load(f)

    return model, le, preprocessor, car_models