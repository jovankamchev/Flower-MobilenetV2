import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)

from .task import (
    get_model, 
    load_data,
    get_parameters, 
    apply_parameters,
    LR,
    WEIGHT_DECAY
)

class SCAFFOLDClient(NumPyClient):
    def __init__(self, partition_id: int, num_partitions: int):
        try:
            self.partition_id = partition_id
            self.num_partitions = num_partitions
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Client {partition_id}: Using device {self.device}")
            
            # Initialize model
            self.model = get_model()
            
            # Initialize client control variate (c_i) - will be set properly on first fit
            self.c_i = None
            
            # Load data
            print(f"Client {partition_id}: Loading data partition...")
            self.train_loader = load_data(partition_id, num_partitions)
            print(f"Client {partition_id}: Data loaded - Train batches: {len(self.train_loader)}")
        except Exception as e:
            print(f"Client {partition_id}: INITIALIZATION ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
    def fit(self, parameters, config):
        """Train the model using SCAFFOLD algorithm."""
        try:
            print(f"Client {self.partition_id}: Starting SCAFFOLD training...")
            
            # Initialize c_i if this is the first round
            if self.c_i is None:
                self.c_i = [np.zeros_like(p) for p in parameters]
                print(f"Client {self.partition_id}: Initialized control variates")
            
            # Initialize c_global (server control variate) - start with zeros
            c_global = [np.zeros_like(p) for p in parameters]
            
            # Store initial parameters (x)
            x_initial = [np.array(p) for p in parameters]
            
            # Apply server parameters to local model
            apply_parameters(self.model, parameters)
            
            # Train model with SCAFFOLD correction
            epochs = config.get("epochs", 1)
            
            print(f"Client {self.partition_id}: Training with epochs={epochs}")
            
            train_loss = self._scaffold_train(
                self.model, 
                self.train_loader, 
                epochs, 
                c_global,
                self.c_i,
                self.device
            )
            
            # Get updated parameters (y)
            y_final = get_parameters(self.model)
            
            # Compute new client control variate (c_i^+)
            # c_i^+ = c_i - c + (x - y) / (K * lr)
            num_batches = len(self.train_loader)
            total_steps = epochs * num_batches
            lr = LR  # Match the learning rate from SGD optimizer
            
            c_i_new = []
            
            for c_i_old, c_g, x, y in zip(self.c_i, c_global, x_initial, y_final):
                # Compute the update
                c_i_plus = c_i_old - c_g + (x - y) / (total_steps * lr)
                c_i_new.append(c_i_plus)
            
            # Compute delta for server aggregation
            c_delta_sum = sum(np.sum(np.abs(new - old)) for new, old in zip(c_i_new, self.c_i))
            
            # Update client control variate
            self.c_i = c_i_new
            
            # Return parameters, number of examples, and metrics
            print(f"Client {self.partition_id}: Training completed - Loss: {train_loss:.4f}, c_delta_sum: {c_delta_sum:.6f}")
            
            return (
                y_final,
                len(self.train_loader.dataset),
                {
                    "train_loss": float(train_loss),
                    "c_delta_sum": float(c_delta_sum)  # Just send a summary metric
                }
            )
        except Exception as e:
            print(f"Client {self.partition_id}: ERROR during training: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _scaffold_train(self, model, train_loader, epochs, c_global, c_i, device):
        """Training loop with SCAFFOLD corrections."""
        model.to(device)
        model.train()

        lr = LR  # Learning rate for SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, momentum=0.9)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Create mapping from parameter names to control variates
        param_names = list(model.state_dict().keys())
        c_i_dict = {name: c_i[i] for i, name in enumerate(param_names)}
        c_global_dict = {name: c_global[i] for i, name in enumerate(param_names)}
        
        epoch_loss = 0.0
        
        for epoch in range(epochs):
            batch_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Apply SCAFFOLD correction to gradients
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None and name in c_i_dict:
                            # Get correction as numpy array
                            correction_np = -c_i_dict[name] + c_global_dict[name]

                            # Convert to tensor with proper shape
                            correction = torch.from_numpy(correction_np).to(device).to(param.dtype)

                            # Add correction to gradient
                            param.grad.add_(correction)
                
                optimizer.step()
                
                batch_loss += loss.item()
            
            epoch_loss = batch_loss / len(train_loader)
            print(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")
        
        return epoch_loss

def client_fn(context: Context):
    """Create and return a SCAFFOLDClient instance."""
    try:
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        
        print(f"client_fn called for partition {partition_id}/{num_partitions}")
        
        client = SCAFFOLDClient(partition_id, num_partitions)
        print(f"SCAFFOLDClient created successfully for partition {partition_id}")
        
        return client.to_client()
    except Exception as e:
        print(f"ERROR in client_fn: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Create the ClientApp
app = ClientApp(client_fn=client_fn)
