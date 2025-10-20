import torch
from typing import List, Tuple, Optional, Dict
from flwr.server import ServerApp, ServerConfig, ServerAppComponents, start_server
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, NDArrays, Scalar
import numpy as np
from collections import defaultdict
import os

from .task import get_model, apply_parameters, get_parameters, load_val, test, MU, EPOCHS

import csv

MODEL_NAME = "best_microf1_model.pt"
CSV_NAME = "best_microf1_logs.csv"

def log_metrics(round_num: int, test_metrics: dict, train_metrics: dict = None, csv_path: str = CSV_NAME):
    """Append test metrics to CSV per training round."""
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        fieldnames = [
            "round", "metric_type",
            "train_loss", "train_accuracy", "train_micro_f1", "train_macro_f1",
            "test_loss", "test_accuracy_exact_match",
            "hamming_loss", "test_micro_f1", "test_macro_f1"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header once
        if not file_exists:
            writer.writeheader()

        # Write test metrics row
        row_data = {
            "round": round_num,
            "metric_type": "test",
            "test_loss": test_metrics["test_loss"],
            "test_accuracy_exact_match": test_metrics["test_accuracy_exact_match"],
            "hamming_loss": test_metrics["hamming_loss"],
            "test_micro_f1": test_metrics["test_micro_f1"],
            "test_macro_f1": test_metrics["test_macro_f1"],
        }
        writer.writerow(row_data)

class FedProxWithMetrics(FedProx):
    """Class defining the FedProx (or any other strategy available in Flower) strategy."""
    
    def __init__(self, mu=MU, csv_path=CSV_FILE, **kwargs):
        super().__init__(proximal_mu=mu, **kwargs)
        self.round_metrics = defaultdict(list)
        self.current_round_train_metrics = {}  # Store train metrics temporarily
        self.csv_path = csv_path
        
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate training results and metrics."""
        print(f"\n=== Round {server_round} - Aggregating Training Results ===")
        
        # Aggregate parameters using parent method
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Collect and average training metrics
        train_metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "train_micro_f1": [],
            "train_macro_f1": [],
            "train_precision_micro": [],
            "train_precision_macro": [],
            "train_recall_micro": [],
            "train_recall_macro": []
        }
        
        total_examples = 0
        for _, fit_res in results:
            total_examples += fit_res.num_examples
            for metric_name in train_metrics.keys():
                if metric_name in fit_res.metrics:
                    train_metrics[metric_name].append(
                        (fit_res.metrics[metric_name], fit_res.num_examples)
                    )
        
        # Calculate weighted averages
        averaged_metrics = {}
        for metric_name, values in train_metrics.items():
            if values:
                weighted_sum = sum(value * weight for value, weight in values)
                averaged_metrics[metric_name] = weighted_sum / total_examples
        
        # Store metrics for this round
        for metric_name, value in averaged_metrics.items():
            self.round_metrics[metric_name].append(value)
        
        # Save training metrics to CSV immediately after aggregation
        self.current_round_train_metrics = averaged_metrics
        self._log_training_metrics(server_round, averaged_metrics)
        
        # Print training metrics
        print(f"Training Metrics (Round {server_round}):")
        for metric_name, value in averaged_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        return aggregated_parameters, aggregated_metrics
    
    def _log_training_metrics(self, server_round: int, train_metrics: dict):
        """Log training metrics to CSV."""
        file_exists = os.path.isfile(self.csv_path)
        
        with open(self.csv_path, mode="a", newline="") as f:
            fieldnames = [
                "round", "metric_type",
                "train_loss", "train_accuracy", "train_micro_f1", "train_macro_f1",
                "test_loss", "test_accuracy_exact_match",
                "hamming_loss", "test_micro_f1", "test_macro_f1"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            # Write training metrics row
            row_data = {
                "round": server_round,
                "metric_type": "train",
                "train_loss": train_metrics.get("train_loss", ""),
                "train_accuracy": train_metrics.get("train_accuracy", ""),
                "train_micro_f1": train_metrics.get("train_micro_f1", ""),
                "train_macro_f1": train_metrics.get("train_macro_f1", ""),
            }
            writer.writerow(row_data)
    
    def print_final_summary(self):
        """Print final training summary."""
        print("\n" + "="*60)
        print("FINAL TRAINING SUMMARY")
        print("="*60)
        
        for metric_name, values in self.round_metrics.items():
            if values:
                print(f"{metric_name}:")
                print(f"  Initial: {values[0]:.4f}")
                print(f"  Final: {values[-1]:.4f}")
                print(f"  Best: {max(values) if 'loss' not in metric_name else min(values):.4f}")
        
        print("="*60)

def get_initial_parameters():
    """Get initial model parameters."""
    model = get_model()
    return ndarrays_to_parameters(get_parameters(model))
    
def get_evaluate_fn(total_round):
    """Returns a server-side evaluation function for a centralized dataset."""
    model = get_model()
    server_test_loader = load_val()
    
    best_f1 = {"micro_f1": 0.0}

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Set model weights from parameters
        apply_parameters(model, parameters)

        # Evaluate the model on the server-side test set
        eval_metrics = test(model, server_test_loader, torch.device('cuda'))
        
        current_f1 = eval_metrics["test_micro_f1"]

        # Print and return evaluation metrics
        print(f"Evaluation completed - Loss: {eval_metrics['test_loss']:.4f}, Accuracy: {eval_metrics['test_accuracy_exact_match']:.4f}")
        
        # Log test metrics to CSV
        log_metrics(server_round, eval_metrics, train_metrics=None)
        

        if current_f1 > best_f1["micro_f1"]:
            best_f1["micro_f1"] = current_f1
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}

            # Save model weights
            torch.save(state_dict, MODEL_NAME)
            print(f"Final model saved as {MODEL_NAME}")
        
        return (
            eval_metrics["test_loss"],
            {   
                "test_loss": eval_metrics["test_loss"],
                "test_hamming_loss": eval_metrics["hamming_loss"],
                "test_accuracy": eval_metrics["test_accuracy_exact_match"],
                "test_micro_f1": eval_metrics["test_micro_f1"],
                "test_macro_f1": eval_metrics["test_macro_f1"]
            }
        )

    return evaluate

def server_fn(context: Context):
    """Create and configure the server."""
    
    # Get configuration
    config = context.run_config
    num_rounds = config.get("num-server-rounds", 10)
    fraction_fit = config.get("fraction-fit", 0.1)
    fraction_evaluate = 0.0 # Disable client-side evaluation
    min_fit_clients = config.get("min-fit-clients", 20)
    min_evaluate_clients = 0 # No clients are needed for evaluation
    min_available_clients = config.get("min-available-clients", 200)
    
    print(f"Server Configuration:")
    print(f"  Rounds: {num_rounds}")
    print(f"  Fraction Fit: {fraction_fit}")
    print(f"  Fraction Evaluate: {fraction_evaluate} (Server-side evaluation enabled)")
    print(f"  Min Fit Clients: {min_fit_clients}")
    print(f"  Min Evaluate Clients: {min_evaluate_clients}")
    print(f"  Min Available Clients: {min_available_clients}")
    
    # Create strategy
    strategy = FedProxWithMetrics(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=get_initial_parameters(),
        on_fit_config_fn=lambda server_round: {
            "epochs": EPOCHS, # Number of local epochs parameter
        },
        evaluate_fn=get_evaluate_fn(num_rounds)
    )

    # Configure server
    server_config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(
        strategy=strategy,
        config=server_config,
    )

# Create the ServerApp
app = ServerApp(server_fn=server_fn)
