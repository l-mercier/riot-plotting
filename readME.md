# Plotting utilities for the R-IoT

## Installation

The scripts in `mubu-riot-plotting/` depend on a small scientific Python stack. The simplest way to install everything is to create the virtual environment at the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you keep `requirements.txt` inside `mubu-riot-plotting/`, use:

```bash
python -m pip install -r mubu-riot-plotting/requirements.txt
```

## Dependencies

The project currently needs:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

If you want other users to reproduce the environment exactly, keep `requirements.txt` updated whenever you add a new third-party import.