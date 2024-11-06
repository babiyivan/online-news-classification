# TUW-NLP2024

This project uses Python 3.12 and [Poetry](https://python-poetry.org/) for dependency management.

## Setup Instructions

1. **Install Python 3.12** (if not already installed) using [pyenv](https://github.com/pyenv/pyenv):
   ```bash
   pyenv install 3.12
   pyenv local 3.12
   ```

2. **Install Poetry**:
   ```bash
   pip install pipx
   python -m pipx ensurepath
   pipx install poetry
   ```

3. **Add New Dependencies**:
   To add a new library, run:
   ```bash
   poetry add <library>
   ```

4. **Install Project Dependencies**:
   To install all required dependencies, run:
   ```bash
   poetry env use 3.12
   poetry install --no-root
   ```

5. **Install Jupyter** (if not already installed):
   If Jupyter is not installed, you can add it with:
   ```bash
   poetry add jupyter
   ```

6. **Activate the Poetry Virtual Environment**:
   To start working in the virtual environment, type:
   ```bash
   poetry shell
   ```

7. **Run the Jupyter Notebook**:
   Navigate to the notebooks directory and start Jupyter:
   ```bash
   cd notebooks
   jupyter notebook
   ```

Now you're all set to work on the project!
