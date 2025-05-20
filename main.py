'''
from src.neo_ga import NEOGA

import os
import pandas as pd
import matplotlib.pyplot as plt
from src.dataset_loader import load_dataset
from src.ga_optimizer import evolve
from src.search_baselines import run_search

HYPERPARAM_SPACE = {
    "learning_rate": (1e-4, 1e-1),
    "batch_size": [32, 64],
    "dropout_rate": (0.1, 0.5),
    "num_hidden_layers": (1, 3),
    "hidden_layer_size": (64, 256),
    "activation": ['relu', 'tanh'],
    "optimizer": ['adam', 'sgd']
}

def save_scores(scores, filepath):
    df = pd.DataFrame(scores)
    df.to_csv(filepath, index=False)
    print(f"💾 Saved: {filepath}")

def plot_results(ga_history, grid_scores, random_scores, plot_path):
    plt.figure(figsize=(10, 6))
    plt.plot([sum([s['accuracy'] for s in gen]) / len(gen) for gen in ga_history], marker='o', label='GA')
    plt.hlines(sum([s['accuracy'] for s in grid_scores]) / len(grid_scores), 0, len(ga_history)-1, colors='r', linestyles='--', label='Grid Search')
    plt.hlines(sum([s['accuracy'] for s in random_scores]) / len(random_scores), 0, len(ga_history)-1, colors='g', linestyles='--', label='Random Search')
    plt.xlabel("Generation")
    plt.ylabel("Validation Accuracy")
    plt.title("Comparison of Optimization Techniques")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Saved plot: {plot_path}")

def plot_all_metrics(ga_history, grid_scores, random_scores, dataset):
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['blue', 'orange', 'green', 'red']

    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 5))
        plt.plot([np.mean([entry[metric] for entry in gen]) for gen in ga_history], label=f"GA - {metric}", color=colors[i])
        plt.hlines(np.mean([s[metric] for s in grid_scores]), 0, len(ga_history)-1, colors='r', linestyles='--', label="Grid Search")
        plt.hlines(np.mean([s[metric] for s in random_scores]), 0, len(ga_history)-1, colors='g', linestyles='--', label="Random Search")
        plt.xlabel("Generation")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} Comparison on {dataset}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_file = f"plots/{dataset.lower()}_{metric}_comparison.png"
        plt.savefig(plot_file)
        print(f"📊 Saved plot: {plot_file}")
        plt.close()

def run_pipeline(dataset='CIFAR10'):
    print(f"📥 Loading {dataset} dataset...")
    train_loader, val_loader, input_dim = load_dataset(dataset)
    output_dim = 10

    print("🚀 Starting GA Optimization...")
    best_chrom, best_score, ga_history = evolve(HYPERPARAM_SPACE, train_loader, val_loader, input_dim, output_dim, generations=5, pop_size=10)
    print(f"✅ Best GA Score: {best_score:.4f}")

    print("🔎 Running Grid Search...")
    grid_scores = run_search(HYPERPARAM_SPACE, train_loader, val_loader, input_dim, output_dim, method='grid')
    print("✅ Grid Search Completed.")

    print("🎲 Running Random Search...")
    random_scores = run_search(HYPERPARAM_SPACE, train_loader, val_loader, input_dim, output_dim, method='random', n_iter=10)
    print("✅ Random Search Completed.")

    result_path = f"results/{dataset.lower()}"
    os.makedirs(result_path, exist_ok=True)
    print("💾 Saving GA results...")
    save_scores([{**entry, 'chromosome': str(entry['chromosome'])} for entry in ga_history[-1]], os.path.join(result_path, 'ga_best.csv'))

    print("💾 Saving Grid Search results...")
    save_scores([{**entry, 'params': str(entry['chromosome'])} for entry in grid_scores], os.path.join(result_path, 'grid_search.csv'))

    print("💾 Saving Random Search results...")
    save_scores([{**entry, 'params': str(entry['chromosome'])} for entry in random_scores], os.path.join(result_path, 'random_search.csv'))

    print("📊 Plotting Accuracy Comparison...")
    plot_path = f"plots/{dataset.lower()}_accuracy_comparison.png"
    plot_results(ga_history, grid_scores, random_scores, plot_path)

    print("📊 Plotting All Metrics...")
    plot_all_metrics(ga_history, grid_scores, random_scores, dataset)

if __name__ == "__main__":
    for dataset in ['CIFAR10']:
        run_pipeline(dataset)


'''
# main.py
from src.neo_ga import NEOGA
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.dataset_loader import load_dataset
from src.search_baselines import run_search

HYPERPARAM_SPACE = {
    "learning_rate": (1e-4, 1e-1),
    "batch_size": [32, 64],
    "dropout_rate": (0.1, 0.5),
    "num_hidden_layers": (1, 3),
    "hidden_layer_size": (64, 256),
    "activation": ['relu', 'tanh'],
    "optimizer": ['adam', 'sgd']
}

def save_scores(scores, filepath):
    df = pd.DataFrame(scores)
    df.to_csv(filepath, index=False)
    print(f"💾 Saved: {filepath}")

def plot_results(log, grid_scores, random_scores, plot_path):
    """Plots NEO best-F1 over generations, with baseline F1 lines."""
    import numpy as np
    plt.figure(figsize=(10, 6))
    gens = list(range(1, len(log['best']) + 1))
    # Plot NEO best F1 per generation
    plt.plot(gens, log['best'], marker='o', label='NEO Best F1')
    # Baseline F1-scores
    grid_f1 = np.mean([s['f1_score'] for s in grid_scores])
    rand_f1 = np.mean([s['f1_score'] for s in random_scores])
    plt.hlines(grid_f1, 1, gens[-1], colors='r', linestyles='--', label='Grid Search F1')
    plt.hlines(rand_f1, 1, gens[-1], colors='g', linestyles='--', label='Random Search F1')
    plt.xlabel("Generation")
    plt.ylabel("F1 Score")
    plt.title("NEO F1 Score Convergence vs Baselines")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Saved F1 convergence plot: {plot_path}")


def run_pipeline(dataset='CIFAR10'):
    print(f"📥 Loading {dataset}...")
    # new
    train_loader, val_loader, test_loader, input_dim, output_dim = load_dataset(dataset)

    output_dim = 10

    print("🚀 Starting NEO Optimization...")
    neo = NEOGA(
        hyperparameter_space=HYPERPARAM_SPACE,
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        output_dim=output_dim,
        pop_size=12,
        generations=25,
        elite_size=3,
        base_mutation_rate=0.15,
        mutation_decay=0.95,
        tournament_k=3,
        eval_workers=1,
        epochs_per_eval=2,
        subset_fraction=0.3,
        early_stop_patience=5,
    )
    best_chrom, best_score, log = neo.evolve()
    print(f"✅ Completed NEO. Best F1: {best_score:.4f}")
    print(f"Best Hyperparameters: {best_chrom}\n")

    print("🔎 Running Grid Search...")
    grid_scores = run_search(
        HYPERPARAM_SPACE, train_loader, val_loader,
        input_dim, output_dim, method='grid'
    )
    print("✅ Grid Search done.\n")

    print("🎲 Running Random Search...")
    random_scores = run_search(
        HYPERPARAM_SPACE, train_loader, val_loader,
        input_dim, output_dim, method='random', n_iter=10
    )
    print("✅ Random Search done.\n")

    # Save GA evolution log
    os.makedirs(f"results/{dataset.lower()}", exist_ok=True)
    log_df = pd.DataFrame({
        'best_f1': log['best'],
        'mean_f1': log['mean'],
        'std_f1': log['std'],
        'elapsed_time': log['time']
    })
    log_path = f"results/{dataset.lower()}/neo_log.csv"
    log_df.to_csv(log_path, index_label='generation')
    print(f"💾 Saved GA log: {log_path}")

    # Save best result
    best_df = pd.DataFrame([{'chromosome': str(best_chrom), 'best_f1': best_score}])
    best_path = f"results/{dataset.lower()}/neo_best.csv"
    best_df.to_csv(best_path, index=False)
    print(f"💾 Saved best GA result: {best_path}")

    # Save baseline results
    save_scores([{**e, 'params': str(e['chromosome'])} for e in grid_scores], f"results/{dataset.lower()}/grid_search.csv")
    save_scores([{**e, 'params': str(e['chromosome'])} for e in random_scores], f"results/{dataset.lower()}/random_search.csv")

        # Plot convergence
    plot_results(log, grid_scores, random_scores, f"plots/{dataset.lower()}_f1_convergence.png")

    # === Final test evaluation for all methods ===
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import torch.nn as nn

    # Load test dataset
    test_transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=best_chrom['batch_size'], shuffle=False)

    def evaluate_model(model, loader, device):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                xb = xb.view(xb.size(0), -1)
                out = model(xb)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='macro'),
            'recall': recall_score(all_labels, all_preds, average='macro'),
            'f1_score': f1_score(all_labels, all_preds, average='macro')
        }

    # Prepare results container
    final_results = []

    # Evaluate NEO best model (few-epoch-trained)
    best_neo_model = neo._build_model(best_chrom).to(neo.device)
    # Optionally retrain on full train+val; here we use existing model
    neo_metrics = evaluate_model(best_neo_model, test_loader, neo.device)
    neo_metrics['method'] = 'NEOGenetic'
    final_results.append(neo_metrics)

    # Evaluate Grid best model
    best_grid = max(grid_scores, key=lambda x: x['f1_score'])
    grid_model = neo._build_model(best_grid['chromosome']).to(neo.device)
    grid_metrics = evaluate_model(grid_model, test_loader, neo.device)
    grid_metrics['method'] = 'GridSearch'
    final_results.append(grid_metrics)

    # Evaluate Random best model
    best_rand = max(random_scores, key=lambda x: x['f1_score'])
    rand_model = neo._build_model(best_rand['chromosome']).to(neo.device)
    rand_metrics = evaluate_model(rand_model, test_loader, neo.device)
    rand_metrics['method'] = 'RandomSearch'
    final_results.append(rand_metrics)

    # Optionally default baseline: simple MLP with mid-range hyperparameters
    default_chrom = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'dropout_rate': 0.2,
        'num_hidden_layers': 1,
        'hidden_sizes': [128],
        'activations': ['relu'],
        'optimizer': 'adam'
    }
    default_model = neo._build_model(default_chrom).to(neo.device)
    default_metrics = evaluate_model(default_model, test_loader, neo.device)
    default_metrics['method'] = 'DefaultMLP'
    final_results.append(default_metrics)

        # Save final test metrics
    metrics_df = pd.DataFrame(final_results)
    metrics_path = f"results/{dataset.lower()}/test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"💾 Saved test metrics: {metrics_path}")

    # === Save best hyperparameters for all methods ===
    best_params = [
        {'method': 'NEOGenetic', 'hyperparameters': str(best_chrom)},
        {'method': 'GridSearch', 'hyperparameters': str(best_grid['chromosome'])},
        {'method': 'RandomSearch', 'hyperparameters': str(best_rand['chromosome'])},
        {'method': 'DefaultMLP', 'hyperparameters': str(default_chrom)}
    ]
    best_params_df = pd.DataFrame(best_params)
    best_params_path = f"results/{dataset.lower()}/best_hyperparameters.csv"
    best_params_df.to_csv(best_params_path, index=False)
    print(f"💾 Saved best hyperparameters: {best_params_path}")

if __name__ == "__main__":
    run_pipeline('CIFAR10')

