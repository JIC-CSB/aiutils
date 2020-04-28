import torch
import dtoolcore

def load_model_weights_from_dataset(model, dataset_uri):
    model_idn = dtoolcore.utils.generate_identifier("model.pt")
    ds = dtoolcore.DataSet.from_uri(dataset_uri)
    state_abspath = ds.item_content_abspath(model_idn)
    model.load_state_dict(torch.load(state_abspath, map_location='cpu'))
    model.eval()

    return model
