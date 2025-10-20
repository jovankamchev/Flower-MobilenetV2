import torch
from typing import List, Tuple, Optional, Dict
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import Strategy
from flwr.common import (
    Context, 
    ndarrays_to_parameters, 
    parameters_to_ndarrays, 
    NDArrays, 
    Scalar,
    FitRes,
    FitIns,
    EvaluateRes,
    EvaluateIns,
    Parameters
)
import numpy as np
from collections import defaultdict
import os
import csv

from .task import get_model, apply_parameters, get_parameters, load_val, test, EPOCHS

MODEL_NAME = "best_microf1_model_scaffold.pt"
CSV_NAME = "best_microf1_logs_scaffold.csv"

def log_metrics(round_num: int, metrics: dict, csv_path: str = CSV_NAME):
    """Append evaluation metrics to CSV per training round."""
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "round", "test_loss", "test_accuracy_exact_match",
            "hamming_loss", "test_micro_f1", "test_macro_f1"
        ])
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "round": round_num,
            "test_loss": metrics["test_loss"],
            "test_accuracy_exact_match": metrics["test_accuracy_exact_match"],
            "hamming_loss": metrics["hamming_loss"],
            "test_micro_f1": metrics["test_micro_f1"],
            "test_macro_f1": metrics["test_macro_f1"],
        })

class SCAFFOLDStrategy(Strategy):
    """SCAFFOLD strategy implementation."""
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 0,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn=None,
        on_fit_config_fn=None,
        num_rounds: int = 10
    ):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.num_rounds = num_rounds
        
        # Initialize server control variate (c)
        if initial_parameters:
            params = parameters_to_ndarrays(initial_parameters)
            self.c_global = [np.zeros_like(p) for p in params]
        else:
            self.c_global = None
        
        self.round_metrics = defaultdict(list)
        self.current_round = 0
    
    def initialize_parameters(self, client_manager):
        """Initialize global model parameters."""
        return self.initial_parameters
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[any, FitIns]]:
        """Configure the next round of training."""
        self.current_round = server_round
        
        # Sample clients
        sample_size = max(
            int(self.fraction_fit * client_manager.num_available()),
            self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients
        )
        
        # Get training config
        config = {}
        if self.on_fit_config_fn:
            config = self.on_fit_config_fn(server_round)
        
        # NOTE: We don't pass c_global through config anymore to avoid serialization issues
        # Clients will use zeros for c_global (standard SCAFFOLD without server control variate)
        
        # Create FitIns
        fit_ins = FitIns(parameters, config)
        
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[any, FitRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results using SCAFFOLD."""
        print(f"\n=== Round {server_round} - Aggregating Training Results (SCAFFOLD) ===")
        
        if not results:
            return None, {}
        
        # Convert results to weights
        weights_results = []
        num_examples_total = 0
        
        for client, fit_res in results:
            weights_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
            num_examples_total += fit_res.num_examples
        
        # Aggregate model parameters (standard weighted averaging)
        aggregated_weights = [
            np.zeros_like(weights_results[0][0][i], dtype=np.float64)
            for i in range(len(weights_results[0][0]))
        ]
        
        for weights, num_examples in weights_results:
            for i, w in enumerate(weights):
                # Cast to float64 to avoid dtype issues
                aggregated_weights[i] += w.astype(np.float64) * (num_examples / num_examples_total)
        
        # Cast back to original dtypes
        aggregated_weights = [
            w.astype(weights_results[0][0][i].dtype) 
            for i, w in enumerate(aggregated_weights)
        ]
        
        # Collect training metrics
        train_losses = []
        for _, fit_res in results:
            if "train_loss" in fit_res.metrics:
                train_losses.append((fit_res.metrics["train_loss"], fit_res.num_examples))
        
        # Calculate weighted average training loss
        avg_train_loss = 0.0
        if train_losses:
            weighted_sum = sum(loss * weight for loss, weight in train_losses)
            avg_train_loss = weighted_sum / num_examples_total
            self.round_metrics["train_loss"].append(avg_train_loss)
            print(f"Training Loss (Round {server_round}): {avg_train_loss:.4f}")
        
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)
        metrics_aggregated = {"train_loss": avg_train_loss}
        
        return parameters_aggregated, metrics_aggregated
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[any, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        
        # Sample clients for evaluation (if needed)
        sample_size = max(
            int(self.fraction_evaluate * client_manager.num_available()),
            self.min_evaluate_clients
        )
        
        if sample_size == 0:
            return []
        
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_evaluate_clients
        )
        
        evaluate_ins = EvaluateIns(parameters, {})
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[any, EvaluateRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}
        
        # This is typically not used with SCAFFOLD as we do server-side evaluation
        return None, {}
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model on server using the evaluate_fn."""
        if self.evaluate_fn is None:
            return None
        
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_result = self.evaluate_fn(server_round, parameters_ndarrays, {})
        
        if eval_result is None:
            return None
        
        loss, metrics = eval_result
        return loss, metrics

def get_initial_parameters():
    """Get initial model parameters."""
    model = get_model()
    return ndarrays_to_parameters(get_parameters(model))
    
def get_evaluate_fn(total_rounds):
    """Returns a server-side evaluation function for a centralized dataset."""
    model = get_model()
    server_test_loader = load_val()
    
    best_f1 = {"micro_f1": 0.0}  # or use "loss": float("inf") if you want to save based on loss


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
        print(f"Evaluation completed - Loss: {eval_metrics['test_loss']:.4f}, "
              f"Accuracy: {eval_metrics['test_accuracy_exact_match']:.4f}")
        
        log_metrics(server_round, eval_metrics)
        
        # Save final model
        if current_f1 > best_f1["micro_f1"]: # server_round != 0 and server_round == total_rounds:
            best_f1["micro_f1"] = current_f1
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
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
    fraction_evaluate = 0.0  # Disable client-side evaluation
    min_fit_clients = config.get("min-fit-clients", 20)
    min_evaluate_clients = 0
    min_available_clients = config.get("min-available-clients", 200)
    
    print(f"Server Configuration (SCAFFOLD):")
    print(f"  Rounds: {num_rounds}")
    print(f"  Fraction Fit: {fraction_fit}")
    print(f"  Fraction Evaluate: {fraction_evaluate} (Server-side evaluation enabled)")
    print(f"  Min Fit Clients: {min_fit_clients}")
    print(f"  Min Evaluate Clients: {min_evaluate_clients}")
    print(f"  Min Available Clients: {min_available_clients}")
    
    # Create SCAFFOLD strategy
    strategy = SCAFFOLDStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=get_initial_parameters(),
        on_fit_config_fn=lambda server_round: {"epochs": EPOCHS},
        evaluate_fn=get_evaluate_fn(num_rounds),
        num_rounds=num_rounds
    )

    # Configure server
    server_config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(
        strategy=strategy,
        config=server_config,
    )

# Create the ServerApp
app = ServerApp(server_fn=server_fn)
