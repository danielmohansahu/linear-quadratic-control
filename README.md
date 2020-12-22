# linear-quadratic-control
Example design and simulation of a Linear Quadratic Controller.

## Dependencies

This repository was tested on an Ubuntu 18.04 machine, but should work for any system capable of running a modern version of Python3.

To install dependencies:

```bash
pip3 install -r requirements.txt
```

## Usage

To run, call the main script via:

```bash
python3 linearize.py
```

This will produce a fair amount of console output demonstrating the system analysis and linearization, and a number of plots simulating the responses of the Controller and Estimator.

