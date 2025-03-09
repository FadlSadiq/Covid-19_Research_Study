# CBF Epidemic Model
## Overview
This code contains an implementation of the Compartmental Behavioral Feedback (CBF) epidemic model - a compartmental model that extends traditional epidemiological models by incorporating human behavioral responses to disease prevalence.
The CBF model introduces a novel compartment (S_B) that represents susceptible individuals who have adopted protective behaviors in response to the epidemic, providing a more realistic simulation of how populations respond to outbreaks.
Model Structure
The model includes the following compartments for each age group:

S: Susceptible individuals
S_B: Behavioral Susceptible individuals (those who have adopted protective measures)
E: Exposed individuals (infected but not yet infectious)
I: Infected individuals
R: Recovered individuals

Features

Age-structured population with customizable age groups
Contact matrices to model interactions between different age groups
Seasonal modulation of transmission rates
Behavioral response dynamics based on epidemic severity
Stochastic transitions between compartments

Key Parameters

beta: Base transmission rate
beta_B: Behavioral change adoption rate
mu_B: Behavior relaxation rate
gamma: Sensitivity to reported infections
r: Efficacy of behavioral changes in reducing transmission
epsilon: Rate of progression from Exposed to Infected
mu: Recovery rate

Seasonal Effects
The model accounts for seasonality with hemisphere-specific effects:

Northern hemisphere: Peak transmission in winter (January 15th)
Southern hemisphere: Peak transmission in winter (July 15th)

Usage Example
pythonCopy# Define population structure and contact patterns
population_by_age = [100, 120, 150, 200, 250, 220, 180, 150, 100, 100]
contact_matrix = np.array([...])  # 10x10 matrix of contact rates

## Initialize model
model = CBFModel(
    population=sum(population_by_age),
    initial_infected=500,
    population_by_age=population_by_age,
    contact_matrix=contact_matrix
)

## Run simulation
days = 100
results = model.simulate(days)

## Visualize results
model.visualize_simulation(results, days)
Visualization
The model provides built-in visualization functionality to display the progression of each compartment over time, helping to understand how behavioral changes affect epidemic dynamics.
Requirements

NumPy
Matplotlib

## Limitations and Future Work

The current implementation uses a simplified approach to model behavioral changes
Future extensions could include:

Vaccination compartments
More complex behavioral dynamics
Economic impact modeling
Healthcare capacity constraints
Geographic spread of disease

## ERROR
THERE IS STILL SOME ERROR IN THE MULTINOMIAL...