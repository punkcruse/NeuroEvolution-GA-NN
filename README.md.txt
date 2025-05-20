
========================================================


"Neuroevolutionary Optimization: A Genetic Algorithm-Based Framework 
for Neural Network Training and Architecture Search"

Directory Structure (NEO Project)
----------------------------------

📁 neo/
├── main.py                # Entry point to launch the NEO training pipeline
├── setup_and_run.py      # Wrapper to configure and execute experiments
├── config/
│   └── hyperparams.json  # Hyperparameter space definitions
├── src/
│   ├── model.py          # EvolvedNN model definition
│   ├── ga.py             # Core genetic algorithm (selection, crossover, mutation)
│   ├── evaluate.py       # Fitness evaluation using brief training
│   ├── search_space.py   # Chromosome encoding logic
│   ├── utils.py          # Misc utilities for logging and plotting
├── results/
│   ├── neo_log.csv       # Generation-wise log of F1-scores and elapsed time
│   ├── neo_best.csv      # Best individuals and their performance
│   ├── plots/
│       ├── mnist_f1_convergence.png
│       ├── cifar10_f1_convergence.png
│       └── accuracy_comparison.png
├── baseline/
│   ├── grid_search.csv   # Baseline grid search results
│   └── random_search.csv # Baseline random search results

Notes:
------
- The `src/` directory contains the core evolutionary algorithm logic.
- The `results/` directory includes performance logs and visual comparisons.
- The `baseline/` folder holds comparative experiments with classical tuning methods.

All code and results have been anonymized for double-blind review.

Refer to `main.py` for entry into the training pipeline and `setup_and_run.py` for reproducibility across datasets.
