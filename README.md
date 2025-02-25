# DR_PINN

Official implementation of the paper "Novel and general 
discontinuity-removing PINNs for elliptic interface problems".(https://doi.org/10.1016/j.jcp.2025.113861)

DR_PINN is a deep learning framework designed for solving elliptic interface problems with high accuracy optimization. The code is organized on modularity, enabling users to easily customize, extend, and integrate its various components into their projects. 

## Key Features

- **Modular Design**: The repository features a modular design for core functionalities such as geometry, model, loss, optimization, data generation, and metrics&visualization, making the codebase easy to extend.
- **Geometry Module**: Compatibly extended from DeepXDE's geometry module, it includes level set function-based geometries (implemented in geometry/geometry_custom.py) and integrates the distmesh algorithm (from pydistmesh repository) for uniform mesh generation.
- **Optimizers Integration**: Comprehensive optimization methods are provided in optimizers.py, including the innovative geodesic-acceleration Levenberg-Marquardt (gd-LM) optimizer, along with standard Levenberg-Marquardt, Adam, and LBFGS.
- **Automatic Differentiation-based Data Generation**: The data_gen.py generates precise training data by leveraging automatic differentiation techniques to accurately compute derivatives.

## Installation

1. Clone the repository, and `cd DR_PINN`


2. Install the required dependencies:
  
   ```bash
   # First, make sure pytorch version > 2.0
   pip install -r requirements.txt
   ```

## License

Due to the utilization of pydistmesh, this project is released under the GNU General Public License v2.0. See the [LICENSE](LICENSE) file for details.
