"""
Industrial Organization: Cournot Competition Analysis
====================================================

This module implements a comprehensive analysis of Cournot competition equilibrium
across multiple markets, including market power estimation and counterfactual analysis.

Key Features:
- Cournot equilibrium computation for N-firm oligopoly markets
- Endogeneity-aware econometric estimation (OLS, 2SLS)
- Market concentration and competition analysis
- Counterfactual policy simulation
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class CournotMarketSimulator:
    """
    Simulates Cournot competition equilibrium across multiple markets.
    
    This class encapsulates the logic for generating synthetic market data,
    solving for Cournot equilibrium, and computing market outcomes.
    """
    
    def __init__(self, n_markets: int = 1000, n_firms: int = 4, seed: int = 42):
        """
        Initialize the market simulator with structural parameters.
        
        Parameters:
        -----------
        n_markets : int
            Number of markets to simulate
        n_firms : int  
            Number of firms per market
        seed : int
            Random seed for reproducibility
        """
        self.M = n_markets
        self.N = n_firms
        self.seed = seed
        
        # Structural parameters (demand)
        self.alpha = {'intercept': 100, 'quantity': -2, 'income': 1.5}
        
        # Structural parameters (supply/marginal cost)
        self.gamma = {'intercept': 5, 'wage': 2, 'productivity': 1}
        
        np.random.seed(seed)
        
    def _generate_exogenous_variables(self) -> Dict[str, np.ndarray]:
        """Generate exogenous market-level variables."""
        return {
            'income': np.random.normal(50, 1, self.M),           # Market income (Ym)
            'wage': np.random.normal(1, 0.25, self.M),           # Wage rate (Wm)  
            'demand_shock': np.random.normal(0, 1, self.M),      # Demand shock (xi_m)
            'cost_shock': np.random.normal(0, 1, (self.M, self.N))  # Firm cost shocks
        }
    
    def _solve_cournot_equilibrium(self, market_idx: int, exog_vars: Dict[str, np.ndarray], 
                                 gamma_intercept: float = None) -> Dict[str, float]:
        """
        Solve for Cournot-Nash equilibrium in a single market.
        
        Parameters:
        -----------
        market_idx : int
            Market index
        exog_vars : dict
            Dictionary of exogenous variables
        gamma_intercept : float, optional
            Override for marginal cost intercept (for counterfactuals)
            
        Returns:
        --------
        dict : Market equilibrium outcomes
        """
        m = market_idx
        gamma_0 = gamma_intercept if gamma_intercept is not None else self.gamma['intercept']
        
        # Marginal cost for each firm
        mc = (gamma_0 + 
              self.gamma['wage'] * exog_vars['wage'][m] + 
              exog_vars['cost_shock'][m, :])
        
        # Cournot equilibrium quantity (derived from FOCs)
        q_equilibrium = ((self.alpha['intercept'] + 
                         self.alpha['income'] * exog_vars['income'][m] + 
                         exog_vars['demand_shock'][m] - mc) / 
                        (2 * self.gamma['productivity'] - 
                         self.alpha['quantity'] * (self.N - 1)))
        
        # Ensure non-negative quantities
        firm_quantities = np.maximum(q_equilibrium, 0)
        total_quantity = np.sum(firm_quantities)
        
        # Market price from inverse demand
        price = (self.alpha['intercept'] + 
                self.alpha['quantity'] * total_quantity + 
                self.alpha['income'] * exog_vars['income'][m] + 
                exog_vars['demand_shock'][m])
        
        # Market structure metrics
        market_shares = firm_quantities / total_quantity if total_quantity > 0 else np.zeros(self.N)
        lerner_indices = (price - mc) / price if price > 0 else np.zeros(self.N)
        
        return {
            'firm_quantities': firm_quantities,
            'total_quantity': total_quantity,
            'price': price,
            'market_shares': market_shares,
            'lerner_market': np.sum(market_shares * lerner_indices),
            'hhi': np.sum(market_shares ** 2),
            'avg_marginal_cost': np.mean(mc)
        }
    
    def simulate_markets(self, gamma_intercept_override: float = None) -> pd.DataFrame:
        """
        Simulate equilibrium outcomes across all markets.
        
        Parameters:
        -----------
        gamma_intercept_override : float, optional
            Override marginal cost intercept for counterfactual analysis
            
        Returns:
        --------
        pd.DataFrame : Market-level equilibrium data
        """
        exog_vars = self._generate_exogenous_variables()
        
        # Storage for equilibrium outcomes
        results = {
            'income': exog_vars['income'],
            'wage': exog_vars['wage'], 
            'demand_shock': exog_vars['demand_shock'],
            'total_quantity': np.zeros(self.M),
            'price': np.zeros(self.M),
            'lerner_index': np.zeros(self.M),
            'hhi': np.zeros(self.M),
            'avg_marginal_cost': np.zeros(self.M)
        }
        
        # Solve equilibrium for each market
        for m in range(self.M):
            equilibrium = self._solve_cournot_equilibrium(
                m, exog_vars, gamma_intercept_override
            )
            
            results['total_quantity'][m] = equilibrium['total_quantity']
            results['price'][m] = equilibrium['price']
            results['lerner_index'][m] = equilibrium['lerner_market']
            results['hhi'][m] = equilibrium['hhi']
            results['avg_marginal_cost'][m] = equilibrium['avg_marginal_cost']
        
        return pd.DataFrame(results)


class EconometricAnalyzer:
    """
    Handles econometric estimation and identification strategies.
    
    This class implements various estimation methods addressing endogeneity
    concerns typical in industrial organization empirical work.
    """
    
    @staticmethod
    def estimate_market_power_relationship(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Estimate relationship between market concentration and market power.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Market-level data
            
        Returns:
        --------
        Regression results for Lerner Index ~ HHI relationship
        """
        X = sm.add_constant(df['hhi'])
        y = df['lerner_index']
        return sm.OLS(y, X).fit(cov_type='HC3')  # Robust standard errors
    
    @staticmethod
    def estimate_demand_ols(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Naive OLS estimation of demand equation (ignoring endogeneity).
        
        Note: This suffers from simultaneity bias as price and quantity 
        are jointly determined in equilibrium.
        """
        X = sm.add_constant(df[['total_quantity', 'income']])
        y = df['price']
        return sm.OLS(y, X).fit(cov_type='HC3')
    
    @staticmethod  
    def estimate_demand_2sls(df: pd.DataFrame) -> IV2SLS:
        """
        Two-stage least squares estimation of demand equation.
        
        Instruments quantity with wage (cost shifter) to address endogeneity.
        Wage affects supply but not demand directly (exclusion restriction).
        """
        try:
            return IV2SLS(
                dependent=df['price'],
                exog=sm.add_constant(df['income']), 
                endog=df['total_quantity'],
                instruments=df['wage']
            ).fit(cov_type='robust')
        except Exception as e:
            print(f"Warning: 2SLS estimation failed with error: {e}")
            print("This may indicate weak instruments or numerical issues.")
            return None
    
    @staticmethod
    def estimate_supply_2sls(df: pd.DataFrame) -> IV2SLS:
        """
        Two-stage least squares estimation of marginal cost equation.
        
        Instruments quantity with income (demand shifter) to address endogeneity.
        """
        try:
            return IV2SLS(
                dependent=df['avg_marginal_cost'],
                exog=sm.add_constant(df['wage']),
                endog=df['total_quantity'], 
                instruments=df['income']
            ).fit(cov_type='robust')
        except Exception as e:
            print(f"Warning: 2SLS estimation failed with error: {e}")
            print("This may indicate weak instruments or numerical issues.")
            return None


def display_summary_statistics(df: pd.DataFrame) -> None:
    """Display formatted summary statistics."""
    print("=" * 60)
    print("MARKET EQUILIBRIUM SUMMARY STATISTICS")
    print("=" * 60)
    
    key_vars = ['price', 'total_quantity', 'hhi', 'lerner_index']
    summary = df[key_vars].describe()
    
    print(summary.round(4))
    print("\n")


def display_regression_results(model: Any, title: str) -> None:
    """Display formatted regression results."""
    print("=" * 60) 
    print(f"{title}")
    print("=" * 60)
    
    # Handle different model types (statsmodels vs linearmodels)
    if hasattr(model, 'summary') and callable(model.summary):
        print(model.summary())
    elif hasattr(model, 'summary'):
        print(model.summary)  # linearmodels IV2SLS case
    else:
        print(model)
    print("\n")


def run_counterfactual_analysis(simulator: CournotMarketSimulator) -> None:
    """
    Perform counterfactual analysis: impact of marginal cost reduction.
    
    This simulates a policy intervention (e.g., productivity improvement,
    input cost reduction) and quantifies equilibrium effects.
    """
    print("=" * 60)
    print("COUNTERFACTUAL ANALYSIS: 50% MARGINAL COST REDUCTION")
    print("=" * 60)
    
    # Baseline equilibrium
    baseline_df = simulator.simulate_markets()
    
    # Counterfactual: reduce marginal cost intercept by 50%
    counterfactual_df = simulator.simulate_markets(
        gamma_intercept_override=simulator.gamma['intercept'] / 2
    )
    
    # Compute percentage changes
    changes = {
        'Average Price': (counterfactual_df['price'].mean() / baseline_df['price'].mean() - 1) * 100,
        'Total Quantity': (counterfactual_df['total_quantity'].mean() / baseline_df['total_quantity'].mean() - 1) * 100,
        'Market Power (Lerner)': (counterfactual_df['lerner_index'].mean() / baseline_df['lerner_index'].mean() - 1) * 100,
        'Market Concentration (HHI)': (counterfactual_df['hhi'].mean() / baseline_df['hhi'].mean() - 1) * 100
    }
    
    print("Policy Impact (% Change from Baseline):")
    print("-" * 40)
    for metric, change in changes.items():
        print(f"{metric:<25}: {change:>8.2f}%")


def main():
    """
    Main execution function implementing complete analysis pipeline.
    """
    # Initialize market simulator
    simulator = CournotMarketSimulator(n_markets=1000, n_firms=4, seed=42)
    
    # Generate baseline market data
    market_data = simulator.simulate_markets()
    
    # Display summary statistics
    display_summary_statistics(market_data)
    
    # Initialize econometric analyzer
    analyzer = EconometricAnalyzer()
    
    # Analysis 1: Market Power and Concentration
    market_power_model = analyzer.estimate_market_power_relationship(market_data)
    display_regression_results(market_power_model, "MARKET POWER ANALYSIS: Lerner Index ~ HHI")
    
    # Analysis 2: Demand Estimation (OLS vs 2SLS comparison)
    demand_ols = analyzer.estimate_demand_ols(market_data)
    display_regression_results(demand_ols, "DEMAND ESTIMATION (OLS - Biased)")
    
    demand_2sls = analyzer.estimate_demand_2sls(market_data)
    if demand_2sls is not None:
        display_regression_results(demand_2sls, "DEMAND ESTIMATION (2SLS - Consistent)")
    
    # Analysis 3: Supply/Marginal Cost Estimation  
    supply_2sls = analyzer.estimate_supply_2sls(market_data)
    if supply_2sls is not None:
        display_regression_results(supply_2sls, "MARGINAL COST ESTIMATION (2SLS)")
    
    # Analysis 4: Counterfactual Policy Analysis
    run_counterfactual_analysis(simulator)


if __name__ == "__main__":
    main()