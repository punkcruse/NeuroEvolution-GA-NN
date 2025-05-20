
from sklearn.model_selection import ParameterGrid, ParameterSampler
from src.utils import train_and_evaluate
from src.model import EvolvedNN
import torch.nn as nn

def create_model(params, input_dim, output_dim):
    layers = []
    in_dim = input_dim
    for size, act in zip(params["hidden_sizes"], params["activations"]):
        layers.append(nn.Linear(in_dim, size))
        layers.append(nn.ReLU() if act == "relu" else nn.Tanh())
        layers.append(nn.Dropout(params["dropout_rate"]))
        in_dim = size
    layers.append(nn.Linear(in_dim, output_dim))
    return EvolvedNN(layers)

def evaluate_search(params_list, train_loader, val_loader, input_dim, output_dim):
    scores = []
    for params in params_list:
        model = create_model(params, input_dim, output_dim)
        acc, prec, rec, f1 = train_and_evaluate(model, train_loader, val_loader, params["learning_rate"], params["optimizer"])
        scores.append({
            "chromosome": params,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
    return scores

def run_search(hyper_space, train_loader, val_loader, input_dim, output_dim, method='grid', n_iter=10):
    param_grid = {
        "learning_rate": [0.001, 0.01],
        "batch_size": [32, 64],
        "dropout_rate": [0.2, 0.3],
        "hidden_sizes": [[128, 64], [256, 128]],
        "activations": [["relu", "relu"], ["tanh", "tanh"]],
        "optimizer": ["adam", "sgd"]
    }
    search_list = list(ParameterGrid(param_grid)) if method == 'grid' else list(ParameterSampler(param_grid, n_iter=n_iter))
    return evaluate_search(search_list, train_loader, val_loader, input_dim, output_dim)
