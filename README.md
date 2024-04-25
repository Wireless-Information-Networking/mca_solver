# Memristor Crossbar Array Solver

Code to produce and visualize the results of 'Memristor Crossbar Array Simulation for Deep Learning Applications'.

<!-- We also include some supplementary material we could not fit in the letter. -->


### Prerequisites
___
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
___
The results from the paper can be replicated by changing the global variable `TESTING` to `False` and running:
```sh
./experiment.sh
```


## License
___
Distributed under the GPLv3 License. See [`COPYING`](./COPYING) for more information.


## Contact
___
Elvis Diaz Machado - elvis.diaz@uab.cat

Project Link: [Wireless-Information-Networking/mca_solver](https://github.com/Wireless-Information-Networking/mca_solver)


## Acknowledgments
___
* PhD Supervisor - [Jose Lopez Vicario](https://github.com/JoseVicarioUAB)
* PhD Advisor - [Antoni Morell Perez](https://espainnova.uab.cat/es/node/42)
