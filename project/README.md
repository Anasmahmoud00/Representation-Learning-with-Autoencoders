# Representation Learning with Autoencoders

This project implements Autoencoders (AE) and Variational Autoencoders (VAE) for representation learning, specifically using the Medical MNIST dataset.

## Project Structure

- **`data/`**: Directory for storing raw and processed datasets.
- **`models/`**: Directory where trained model weights (checkpoints) are saved.
- **`notebooks/`**: Contains Jupyter Notebooks for experimentation.
  - `DSAI490_Assignment1_Runner.ipynb`: The main notebook used to run experiments and visualize results.
- **`src/`**: The core source code for the project.
  - `data_processing.py`: Contains functions for downloading, loading, and preprocessing the Medical MNIST data.
  - `model.py`: Defines the neural network architectures for both the Autoencoder and the Variational Autoencoder.
  - `train.py`: Contains the training loops and evaluation logic for the models.
- **`tests/`**: Contains unit tests to ensure code quality.
  - `test_data_processing.py`: Tests for data loading functions.
  - `test_model.py`: Tests for the model architectures.
- **`requirements.txt`**: Lists all the Python packages required to run the project.
- **`README.md`**: This file, explaining the project setup and file structure.

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main experiments using the notebook in `notebooks/`.
