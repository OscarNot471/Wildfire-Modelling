<h1 align="center">Wildfire Modelling</h1>

<p align="center">
  <img src="results/wildfire_3d_animation.gif" width="750">
</p>


## Project Background

This project was developed as part of the course Introduction to Modeling in Physics during my undergraduate degree. At the end of the semester, we were asked to work in groups of three to model a physical system of our choice and present it to the class.

Together with two colleagues, we chose to model wildfire propagation. The main motivation was that wildfire spread can be described, at a basic level, by equations similar to diffusion equations, which we had already studied and implemented in previous practical sessions. This made it a natural and interesting extension of the material covered in the course.


## Modelling Approach 

Our initial intention was to implement the well-known Rothermel model, which is one of the most realistic and widely used wildfire spread models. However, we decided not to pursue this approach because it relies on a large number of experimentally derived parameters that are not straightforward to interpret or estimate. This complexity would have shifted the focus away from the numerical modeling aspects we wanted to explore.

Instead, we developed a simplified physics-based model. The goal was not to achieve high physical realism, but to construct a clear and numerically consistent framework capable of qualitatively reproducing wildfire propagation.

Rather than using many experimentally calibrated parameters, we introduced a smaller set of generalized parameters and normalized fields (for example, moisture ranging from 0 to 1). This approach does not attempt to reproduce the detailed physical properties of real forests or combustion processes. However, it allows us to study how different environmental factors influence fire spread in a controlled and interpretable way.


## Numerical Method

The core of the model is a generalized heat equation including:
* Diffusion
* Advection (wind and slope effects)
* A combustion source term
* Cooling toward ambient temperature

The equation is solved using a Crank–Nicolson scheme on a hexagonal grid. The hexagonal discretization provides a more isotropic neighborhood structure compared to a square grid.

The model was developed progressively:
1. We first implemented the diffusion-based formulation.
2. We then introduced combustion and fuel consumption.
3. Environmental factors were added one by one:
   * Elevation field (Gaussian mountain)
   * Slope-dependent advection
   * Wind field
   * Moisture field
   * Ambient temperature variations depending on elevation and orientation

This step-by-step development allowed us to clearly understand the impact of each physical mechanism on the overall dynamics.

All mathematical formulations are provided in the /docs folder, and the code is commented for clarity.


## Project Contributions

This repository contains:
* The full Python implementation of the diffusion–advection–combustion model (/src)
* The presentation slides used during the oral exposition (/docs)
* Simulation results and animations

Another member of the group implemented an alternative approach based on a Metropolis algorithm. That method is not included in this repository; here I present only the deterministic PDE-based model developed collaboratively by David and myself.


## Results

Although the model is not intended to reproduce real wildfire behavior quantitatively, it is fully functional and dynamically consistent.

The simulations respond coherently to variations in:
* Initial temperature
* Wind intensity and direction
* Elevation and slope
* Vegetation properties
* Moisture levels
* Vegetation distribution

In this sense, the model provides a qualitative representation of wildfire propagation and serves as a solid example of physics-based numerical modeling.
