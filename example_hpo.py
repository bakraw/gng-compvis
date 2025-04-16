"""
Example of how Optuna can be used for hyperparameter optimization.
Requires the optuna package (optuna-dashboard also recommended).
"""

import optuna
import torch
import torchvision
import gng


def objective(trial):
    """
    Objective function for Optuna.
    Doesn't get called by us directly, but by Optuna (see below).

    This is basically a wrapper around example_mnist.py,
    except we specify ranges for the hyperparameters instead of hard-coding them.

    :param trial: Optuna trial object. No need to worry about it.
    """

    # Create dataloader    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.GaussianBlur(3),
        torchvision.transforms.Resize((14, 14)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    training_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testing_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=1000, shuffle=True)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=1000, shuffle=True)

    # Define the hyperparameters to optimize.
    params = {
        "e_b": trial.suggest_float("e_b", 0.001, 1),
        "e_n": trial.suggest_float("e_n", 0.001, 0.1),
        "a_max": trial.suggest_int("a_max", 5, 200),
        "l": trial.suggest_int("l", 5, 100),
        "a": trial.suggest_float("a", 0.01, 1.0),
        "d": trial.suggest_float("d", 0.01, 1.0),
        "passes": trial.suggest_int("passes", 1, 2),
        "max_nodes": trial.suggest_int("max_nodes", 100, 1500)
    }    

    input_dim = 14*14
    device = "cpu" # if params["max_nodes"] < 1000 else "cuda"

    # Create the model, test it and return the measured accuracy.
    model = gng.Gng(input_dim=input_dim, **params, device=device)
    model.train(training_dataloader, 10)
    return model.test(testing_dataloader) 


# Create the study and optimize the objective function.
study = optuna.create_study(direction="maximize", storage="sqlite:///db.sqlite3")
study.optimize(objective, n_trials=200)

