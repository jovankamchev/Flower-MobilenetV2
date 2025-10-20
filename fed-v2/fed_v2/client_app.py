import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from .task import (
    get_model, 
    load_data_random,
    load_data_userid,
    load_data_primary_label,
    load_val,
    train, 
    test, 
    get_parameters, 
    apply_parameters
)

class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int, num_partitions: int):
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Client {partition_id}: Using device {self.device}")
        
        # Initialize model
        self.model = get_model()
        
        # Load data
        print(f"Client {partition_id}: Loading data partition...")
        self.train_loader = load_data_userid(partition_id, num_partitions) # Set the desired data loading function

        print(f"Client {partition_id}: Data loaded - Train batches: {len(self.train_loader)}")
        
    def fit(self, parameters, config):
        """Train the model on the client's data."""
        print(f"Client {self.partition_id}: Starting training...")
        
        # Apply server parameters to local model
        apply_parameters(self.model, parameters)
        
        # Train model
        epochs = config.get("epochs", 1)
        train_loss = train(self.model, self.train_loader, epochs, self.device)
        
        # Get updated parameters
        updated_parameters = get_parameters(self.model)
        
        # Return parameters and metrics
        print(f"Client {self.partition_id}: Training completed - Loss: {train_loss:.4f}")
        
        return (
            updated_parameters,
            len(self.train_loader.dataset),
            {
                "train_loss": train_loss
            }
        )

def client_fn(context: Context):
    """Create and return a FlowerClient instance."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    return FlowerClient(partition_id, num_partitions).to_client()

# Create the ClientApp
app = ClientApp(client_fn=client_fn)
