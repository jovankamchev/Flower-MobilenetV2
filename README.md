
# Federated Fine-Tuning of MobileNetV2 with Flower
## Project Overview: 
This repository hosts a demonstration of Federated Learning (FL) using the Flower framework for fine-tuning a pre-trained MobileNetV2 model for image classification on the Apple FLAIR dataset. The goal is to evaluate the performance of a compact model (such as MobileNetV2) with various federated aggregation strategies (FedAvg, FedProx, SCAFFOLD) under different levels of data non-IIDness (synthetic partitions, original partitions). Two directories are provided, one using the aggregation strategies available in Flower (fed-v2), and another with a custom SCAFFOLD implementation (fed-scaffold).
The experiments were ran in WSL2 using the Flower Simulation Engine and PyTorch.

## Install dependencies and project

The dependencies for each approach (dedicated and custom strategies) are listed in their respective  `pyproject.toml` files and can be installed as follows:

```bash
pip install -e .
```
Note that the same virtual environment was and can be used for both approaches, since they use the same dependencies.
## Getting Started

This project utilizes the Flower Simulation Engine. In the `fed-v2`/`fed-scaffold` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```
The general config file for the simulation parameters (number of clients, federated rounds) is `pyproject.toml`. Training, partitioning and aggregation hyperparameters can be set in the `task.py` file.
NOTE: If ran with a GPU, set the `options.backend.client-resources.num-gpus` accordingly in order to avoid OOM errors (1/num of clients running on GPU concurrently)

### Project Structure
The predefined project structure and needed files are automatically generated when running the command :
```bash
flwr new
```
Then, we define our code in the respective server, client and task Python files.
```
├─ fed-v2             # Scripts which utilize the strategies available from Flower (FedAvg,FedProx)
│	├─── fed_v2
│	│	├─ client_app.py	# Code for the client-side simulation (training)
│	│	├─ server_app.py	# Code for the server-side simulation (aggregation, evaluation)
│	│	└─ task.py	# General functions
│	└─ pyproject.toml	# Configuration and dependencies file
├─── fed-scaffold             # Scripts which utilize SCAFFOLD strategy (custom)
│	└─── fed_scaffold
│	│	├─ client_app.py # Code for the client-side simulation (training)
│	│	├─ server_app.py # Code for the server-side simulation (aggregation, evaluation)
│	│	└─ task.py # General functions
│	└─ pyproject.toml # Configuration and dependencies file
```
## Resources

- [Apple FLAIR Dataset](https://github.com/apple/ml-flair)
- [Flower Framework](https://flower.ai/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [FedProx](https://arxiv.org/abs/1812.06127)
- [SCAFFOLD](https://arxiv.org/abs/1910.06378)
## License
This project is licensed under the MIT License - see the 'LICENSE' file for details.

