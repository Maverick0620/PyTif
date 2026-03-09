<p align="center">
  <img src="https://img.shields.io/github/stars/Luooops/PyTif?style=for-the-badge&logo=github&color=24292e" />
  <img src="https://img.shields.io/github/forks/Luooops/PyTif?style=for-the-badge&logo=github&color=24292e" />
  <img src="https://img.shields.io/github/license/Luooops/PyTif?style=for-the-badge&logo=github&color=24292e" />
</p>

# PyTif
A visualization app for tiff files with pyQt.

## Installation

### Prerequisites
For Windows systems, you need to install the Microsoft Visual C++ Redistributable runtimes. I recommend installing the "All-in-One" package from [TechPowerUp](https://www.techpowerup.com/download/visual-c-redistributable-runtime-package-all-in-one/).

### Setting Up
This project uses `uv` for environment management. To set up the environment and install dependencies, run:

```bash
uv sync
```

## Running the App
To run the program, use:

```bash
uv run src/main.py
```

## Building the App
To build the standalone executable, run:

```bash
uv run pyinstaller PyTif.spec
```

