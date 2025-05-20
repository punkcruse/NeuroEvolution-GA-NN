
import random
from src.utils import train_and_evaluate, save_checkpoint
from src.model import EvolvedNN
import torch.nn as nn

def create_model_from_chromosome(chrom, input_dim, output_dim):
    layers = []
    in_dim = input_dim
    for size, act in zip(chrom["hidden_sizes"], chrom["activations"]):
        layers.append(nn.Linear(in_dim, size))
        layers.append(nn.ReLU() if act == "relu" else nn.Tanh())
        layers.append(nn.Dropout(chrom["dropout_rate"]))
        in_dim = size
    layers.append(nn.Linear(in_dim, output_dim))
    return EvolvedNN(layers)

def evolve(hyper_space, train_loader, val_loader, input_dim, output_dim, generations=5, pop_size=10, save_dir="checkpoints"):
    def init_chrom():
        num_layers = random.randint(*hyper_space["num_hidden_layers"])
        return {
            "learning_rate": round(random.uniform(*hyper_space["learning_rate"]), 5),
            "batch_size": random.choice(hyper_space["batch_size"]),
            "dropout_rate": round(random.uniform(*hyper_space["dropout_rate"]), 2),
            "num_hidden_layers": num_layers,
            "hidden_sizes": [random.randint(*hyper_space["hidden_layer_size"]) for _ in range(num_layers)],
            "activations": [random.choice(hyper_space["activation"]) for _ in range(num_layers)],
            "optimizer": random.choice(hyper_space["optimizer"])
        }

    def crossover(p1, p2):
        child = {}
        for key in p1:
            if key in p2:
                child[key] = random.choice([p1[key], p2[key]])
        else:
            child[key] = p1[key]  # fallback
        return child

    def mutate(chrom):
        key = random.choice(list(hyper_space.keys()))
        if isinstance(hyper_space[key], tuple):
            chrom[key] = round(random.uniform(*hyper_space[key]), 5)
        elif isinstance(hyper_space[key], list):
            chrom[key] = random.choice(hyper_space[key])
        return chrom

    population = [init_chrom() for _ in range(pop_size)]
    best_score = 0
    best_chrom = None
    history = []

    for gen in range(generations):
        print(f"📈 Generation {gen + 1}")
        scores = []
        for idx, chrom in enumerate(population):
            model = create_model_from_chromosome(chrom, input_dim, output_dim)
            acc, prec, rec, f1 = train_and_evaluate(model, train_loader, val_loader, chrom["learning_rate"], chrom["optimizer"])
            print(f"{chrom} -> Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            scores.append({"chromosome": chrom, "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})
            if acc > best_score:
                best_score = acc
                best_chrom = chrom
                save_checkpoint(model, f"{save_dir}/best_model_gen{gen+1}_idx{idx}.pt")
        history.append(scores)

        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population, 2)
            child = crossover(p1, p2)
            if random.random() < 0.3:
                child = mutate(child)
            new_pop.append(child)
        population = new_pop

    return best_chrom, best_score, history