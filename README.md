# Cournot Competition Analysis and Econometric Estimation

## Overview

This repository implements a comprehensive econometric and simulation analysis of Cournot competition equilibrium, widely used in industrial organization to understand market power, competition, and the impacts of market structures. It includes equilibrium simulation, market concentration analysis, econometric estimation addressing endogeneity, and counterfactual policy evaluations.

## Motivation

Quantitative researchers and traders utilize economic modeling and econometric estimation to forecast market behavior and policy impacts. Cournot models, specifically, offer insights into oligopolistic competition, price-setting behaviors, and the strategic interactions of market participants. Demonstrating proficiency in these methods highlights strong analytical, econometric, and computational skills highly valued in quantitative roles.

## Project Components

The project is structured around two main components:

1. **Market Simulation (`cournot.py`)**:
   - Simulates Cournot-Nash equilibria in oligopolistic markets.
   - Generates synthetic data across multiple market scenarios.
   - Computes equilibrium prices, quantities, Lerner indices, and market concentration metrics (HHI).

2. **Econometric Analysis**:
   - Estimates market power relationships using Ordinary Least Squares (OLS) and Two-Stage Least Squares (2SLS).
   - Addresses endogeneity issues in demand and supply estimation.
   - Conducts rigorous regression analyses with robust standard errors.

## Methodologies Used

- **Cournot-Nash Equilibrium Simulation**: Models strategic firm interactions and computes equilibrium outcomes analytically.
- **Instrumental Variables (IV) and 2SLS Estimation**: Corrects for simultaneity and endogeneity bias in demand and supply regressions.
- **Market Concentration Analysis**: Uses Herfindahl-Hirschman Index (HHI) and Lerner Index to quantify market power and competition.
- **Counterfactual Policy Analysis**: Evaluates hypothetical policy impacts on market equilibria through marginal cost modifications.

## Implementation Highlights

- **Robust Data Generation**: Synthesizes realistic market scenarios with carefully calibrated stochastic processes.
- **Econometric Rigor**: Implements advanced econometric methods ensuring unbiased and consistent parameter estimates.
- **Comprehensive Analysis Pipeline**: From data simulation to detailed econometric analysis and policy evaluation.
- **Counterfactual Evaluations**: Clearly quantifies policy impacts, enhancing decision-making and forecasting accuracy.

## Key Results

The analysis provides:

- Precise estimates of market equilibrium outcomes (price, quantity, market power indicators).
- Robust identification of the relationship between market concentration (HHI) and market power (Lerner Index).
- Demonstration of significant differences between naive OLS and unbiased IV (2SLS) estimations.
- Quantitative assessment of counterfactual policies (e.g., a 50% marginal cost reduction), illustrating potential market impacts.

## Repository Structure

```
│
├── cournot.py                   # Main script for Cournot market simulation and analysis
└── README.md                    # Project documentation
```

## Prerequisites

- Python 3.8+
- Required Python packages:
  - numpy, pandas
  - statsmodels
  - linearmodels


## Relevance to Quantitative Roles

This project emphasizes:
- Strong skills in econometric modeling and statistical inference.
- Capability in handling complex simulation tasks for market forecasting.
- Experience addressing endogeneity, crucial in real-world financial and economic analyses.
- Proficiency in conducting counterfactual simulations relevant for strategic decision-making and policy assessment.

## Contact

For inquiries, further discussions, or collaborations:
- Email: j.danielson@mail.utoronto.ca
- LinkedIn: https://www.linkedin.com/in/jasperdanielson/

---

Developed by Jasper Danielson, MA Economics (Finance), University of Toronto
