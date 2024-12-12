# Model-based safe exploration of dynamical systems

## Abstract  

As reinforcement learning (RL) progresses from simulated environments to real-world physical systems, ensuring safety during the learning process becomes critical. Safe exploration addresses the challenge of interacting effectively in an unknown environment while avoiding undesirable outcomes.

This project proposes a model-based RL approach for safe exploration using pessimism to guide the exploration process under model uncertainty. We introduce a stochastic optimization algorithm to solve the pessimistic objective, provide theoretical safety guarantees, and demonstrate its effectiveness on low-dimensional control tasks with safety constraints.

## Contribution  

This work presents a novel algorithm for safe exploration of dynamical systems with unknown dynamics, leveraging pessimism as a guiding principle. This approach assumes a fully observable system with known reward and cost functions.

Key contributions include:

- Reformulating safety objectives under the Constrained Markov Decision Process (CMDP) framework to incorporate pessimism.  
- Developing a stochastic optimization algorithm to solve the reformulated objective.  
- Proving a lower-bound guarantee of safety for the true system under well-calibrated dynamics models.  
- Evaluating performance on continuous control tasks with safety objectives, comparing results to baselines that do not use pessimism.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Conda (optional)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rischiboy/safe_exploration.git
   cd safe_exploration
   
2. Create virtual environment:
  
   ```bash
   conda create --name safex python=3.11

   python -m venv safex

3. Install packages:

   ```bash
   pip install -r requirements.txt

### Setting Up Weights & Biases (wandb)

This project uses Weights & Biases for experiment tracking and visualization. Follow these steps to set up W&B before running any scripts:  

1. **Sign up for W&B**:  
   If you don’t have an account, sign up for free at [wandb.ai](https://wandb.ai).  

2. **Log in to W&B**:  
   After signing up, log in from the command line:  

   ```bash
   wandb login


## Experiments

To run experiments, start by updating the configuration file located in the `./experiments/config` directory for the experiment you want to run. Then, use the launcher script corresponding to the experiment from the `./experiments` directory to execute it. In the launcher script, you’ll also need to specify the optimization method and model to use. The script will combine the configuration file with the selected method and model to launch the experiment.

### Steps to Run an Experiment  

1. **Edit the Configuration File**:  
   Locate the configuration file for your desired experiment in the `./experiments/config` directory. For example:  
   - `./experiments/config/pendulum/minmax_bnn_pendulum.yaml`  

   Open the file and modify the parameters as needed, such as:  
   - **Wandb parameters**:  
     - `logging_wandb: True` – Enable this to log metrics on Weights & Biases.  
   - **Model parameters**:  
     - Define `optimizer` and `agent` to specify the optimization method and the respective model-based agent.  
   - **Training parameters**:  
     - Adjust `epochs` and `batch_size` to set the number of training iterations and batch size for training.  

2. **Configure the Job Scheduler**:  
   You can control the number of experiments to run by adjusting the num_seeds_per_param parameter in the configuration file. All experiments will use the same specified configuration but will run with different seeds. These experiments are grouped together, allowing the metrics tracked during the learning process to be aggregated on Weights & Biases for better comparison and analysis.

   - Configure the number of experiments under `config/job/pendulum_job.yaml` using the `num_seeds_per_hparam` parameter.  
   - Set additional constraints like `num_cpus`, `mem` and `time` in the same file to control resource usage.

3. **Run the Launcher Script**:  
   Use the launcher script corresponding to the experiment in the `./experiments` directory. Specify the configuration file, optimization method, and model. For example:

   ```bash
   python ./experiments/launcher_cem_pendulum_test.py
