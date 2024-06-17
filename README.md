# Memristor Crossbar Array Solver
Code to produce and visualize the results of 'Memristor Crossbar Array Simulation for Deep Learning Applications' (DOI: [10.1109/TNANO.2024.3415382](https://doi.org/10.1109/TNANO.2024.3415382)).

We also include some supplementary material, with an example and additional mathematics about the solver.


### Prerequisites
1. Clone the repo & move into the new directory
```sh
git clone https://github.com/Wireless-Information-Networking/mca_solver.git
cd mca_solver
```

2. Create virtual environment & activate it
```
python -m venv mca_solver
source ./mca_solver/bin/activate
```

3. Install dependencies
```sh
pip install -r requirements.txt
```
Only `torch`, `numpy`, and `scipy` are needed to reproduce the results. 
The `latex` package needs `texlive` to run.


## Usage
The results from the paper can be replicated by changing the global variable `TESTING` to `False` and running:
```sh
./experiment.sh
```


## License
Distributed under the GPLv3 License. See [`COPYING`](./COPYING) for more information.


## Contact
Elvis Diaz Machado - elvis.diaz@uab.cat

Project Link: [Wireless-Information-Networking/mca_solver](https://github.com/Wireless-Information-Networking/mca_solver)


## Acknowledgments
* PhD Supervisor - [Jose Lopez Vicario](https://github.com/JoseVicarioUAB)
* PhD Supervisor - [Antoni Morell Perez](https://github.com/amorell8)
