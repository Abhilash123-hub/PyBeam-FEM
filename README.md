# PyBeam-FEM: Computational Mechanics & ML Predictive Pipeline 🚀

## Overview
PyBeam-FEM is a purely software-based Finite Element Method (FEM) solver designed to analyze various structural beam configurations. It bridges traditional solid mechanics with modern data science techniques. Developed from scratch in Python, this tool parses structural geometry, applies boundary conditions, and computes displacement and stress matrices. 

Beyond classical computation, this project integrates a **Machine Learning module**. By automating thousands of structural simulations, the software generates a robust synthetic dataset used to train a predictive regression model, offering a computationally efficient alternative to iterative FEM processing.

## Key Features
* **Automated Pre-Processing:** Decoupled JSON architecture for defining nodes, elements, boundary conditions, and loads.
* **Custom FEM Engine:** Assembles global stiffness matrices from local elemental data and solves linear systems for multi-type beams (cantilever, continuous, simply supported).
* **Mathematical Visualization:** Uses Hermite shape functions to plot mathematically accurate structural deformation curves.
* **AI/ML Integration:** Generates synthetic FEM data and trains a Random Forest Regressor to instantly predict maximum deflection and structural failure without running the heavy matrix math.

## File Structure
* `beam_config.json`: The structural input parameters.
* `preprocessor.py`: Parses the JSON geometry and load data.
* `solver.py`: The core computational engine (constructs $K$ matrices and solves $[K]\{u\} = \{f\}$).
* `postprocessor.py`: Uses `matplotlib` to render the beam's original and deformed shapes.
* `ml_pipeline.py`: Automates synthetic data generation and trains the AI predictive model.
* `synthetic_beam_data.csv`: The generated dataset.

## Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Abhilash123-hub/PyBeam-FEM.git](https://github.com/Abhilash123-hub/PyBeam-FEM.git)