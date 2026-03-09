[![Repository Card](https://github-readme-stats.vercel.app/api/pin/?username=Luooops&repo=PyTif&show_icons=true&theme=dark)](https://github.com/Luooops/PyTif)

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

