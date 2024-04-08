import os
from dataclasses import make_dataclass
from backend.core.parsers import parse_origins_list


def get_config():
    """load environment variables and return a config object"""

    # for development, try to load .env
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    _config = dict()

    # cors origins
    _config["cors_allow_origins"] = parse_origins_list(str(os.environ.get("CORS_ALLOW_ORIGINS", "")))
    _config["cors_allow_credentials"] = bool(int(os.environ.get("CORS_ALLOW_CREDENTIALS", 0)))

    # following is for accessing swagger from eks
    _config["root_path"] = str(os.environ.get("ROOT_PATH", ""))
    _config["server_path"] = str(os.environ.get("SERVER_PATH", ""))
    _config["openapi_url"] = str(os.environ.get("OPENAPI_URL", "/openapi.json"))
    _config["swagger_url"] = str(os.environ.get("SWAGGER_URL", "/documentation"))

    # custom parameters
    _config["model_path"] = str(os.environ.get("MODEL_PATH", "/models/model.pkl"))
    _config["one_hot_encoder_path"] = str(os.environ.get("ONE_HOT_ENCODER_PATH", "/models/one_hot_encoder.pkl"))
    _config["label_encoder_path"] = str(os.environ.get("LABEL_ENCODER_PATH", "/models/label_encoder.pkl"))
    _config["data_path"] = str(os.environ.get("DATA_PATH", "/data/data.csv"))

    # make dataclass to access these variables
    Config = make_dataclass("Config", fields=[(k, type(v)) for k, v in _config.items()])
    config = Config(**_config)

    return config


config = get_config()
