import importlib


def get_model(config, dataset):
    model_name = config['model']
    model_module = importlib.import_module("myrec.models." + model_name)
    model_class = getattr(model_module, model_name)
    return model_class(config, dataset).get_model()
