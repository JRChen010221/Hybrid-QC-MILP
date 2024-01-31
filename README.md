# Info

The project is a replication of the paper "Hybrid Classical-Quantum Optimization Techniques for Solving Mixed-Integer Programming Problems in Production Scheduling" by A. Ajagekar and F. You published on "IEEE Transactions on Quantum Engineering" 2022.

## Requirements

Use the `pip` command to install the dependencies.
```
pip install -r requirements.txt
```

## Usage

Run the `main.py` file along with some command line arguments to specify the type of solver.
```
python main.py --job_num --machine_num --task [classic | hybrid] --device [sim | real]
```
`--device`is only specified for the use of either simulated annealing sampler of real quantum annealing sampler

