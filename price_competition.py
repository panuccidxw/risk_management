import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve


# Load and prepare sample pricing data
def calculate_price_elasticity(df):
    """
    Calculate price elasticity using log-log regression:
    Why log-log regression? ln(Q) = alpha + beta * ln(P) + epsilon
    take derivative above equation --> d(ln(Q)) / d(ln(P)) = beta
    because d(ln(x)) / dx = 1/x, (dQ/Q) / (dP/P) = beta --> exactly the definition of elasticity
    """
    # Log transformation for elasticity calculation
    df['log_price'] = np.log(df['AveragePrice'])
    df['log_sales'] = np.log(df['UnitsSold'])

    # Remove outliers using Z-score
    z_scores = np.abs((df['log_price'] - df['log_price'].mean()) / df['log_price'].std())
    df_clean = df[z_scores < 3]

    # Linear regression: log(sales) = α + β*log(price)
    X = df_clean['log_price']
    Y = df_clean['log_sales']
    X = sm.add_constant(X)

    model = sm.OLS(Y, X)
    results = model.fit()

    # β coefficient is the price elasticity
    elasticity = results.params[1]
    p_value = results.pvalues[1]

    print(f"Price Elasticity: {elasticity:.2f}")
    print(f"P-value: {p_value:.4f}")

    return elasticity, results


def calculate_cross_elasticity(df_product_a, df_product_b):
    """
    Calculate cross-price elasticity between two competing products
    (dQ_a/Q_a) / (dP_b/P_b)
    """
    # Merge data on time periods
    df_merged = pd.merge(df_product_a, df_product_b, on='time_period', suffixes=('_a', '_b'))

    # Log transformation
    df_merged['log_price_b'] = np.log(df_merged['AveragePrice_b'])
    df_merged['log_sales_a'] = np.log(df_merged['UnitsSold_a'])

    # Regression: log(sales_A) = α + β*log(price_B)
    X = df_merged['log_price_b']
    Y = df_merged['log_sales_a']
    X = sm.add_constant(X)

    model = sm.OLS(Y, X)
    results = model.fit()

    cross_elasticity = results.params[1]

    print(f"Cross Elasticity: {cross_elasticity:.2f}")

    return cross_elasticity


# Example data generation (replace with real data)
np.random.seed(42)
time_periods = range(1, 101)

# Product 1 data
product1_data = {
    'time_period': time_periods,
    'AveragePrice': 50 + np.random.normal(0, 5, 100) + 0.1 * np.arange(100),
    'UnitsSold': 1000 - 15 * (50 + np.random.normal(0, 5, 100)) + np.random.normal(0, 50, 100)
}
df_product1 = pd.DataFrame(product1_data)
df_product1['UnitsSold'] = np.maximum(df_product1['UnitsSold'], 100)  # Ensure positive sales

# Product 2 data
product2_data = {
    'time_period': time_periods,
    'AveragePrice': 60 + np.random.normal(0, 6, 100) + 0.05 * np.arange(100),
    'UnitsSold': 800 - 12 * (60 + np.random.normal(0, 6, 100)) + np.random.normal(0, 40, 100)
}
df_product2 = pd.DataFrame(product2_data)
df_product2['UnitsSold'] = np.maximum(df_product2['UnitsSold'], 80)

# Calculate Four elasticities
print("=== PRODUCT 1 PRICE ELASTICITY ===")
elasticity_1, _ = calculate_price_elasticity(df_product1)

print("\n=== PRODUCT 2 PRICE ELASTICITY ===")
elasticity_2, _ = calculate_price_elasticity(df_product2)

print("\n=== CROSS ELASTICITIES ===")
cross_elasticity_1_2 = calculate_cross_elasticity(df_product1, df_product2)
cross_elasticity_2_1 = calculate_cross_elasticity(df_product2, df_product1)


def calculate_profit_change(price_change_self, price_change_competitor,
                            own_elasticity, cross_elasticity,
                            base_sales, base_price):
    """
    Calculate percentage profit change given price changes

    Args:
        price_change_self: Price change for this product ($)
        price_change_competitor: Competitor's price change ($)
        own_elasticity: Own price elasticity
        cross_elasticity: Cross-price elasticity
        base_sales: Current sales volume
        base_price: Current price

    Returns:
        Percentage change in profit
    """
    # New price
    new_price = base_price + price_change_self

    # Demand change due to own price change
    own_demand_change = own_elasticity * (price_change_self / base_price) * base_sales

    # Demand change due to competitor price change
    cross_demand_change = cross_elasticity * (price_change_competitor / base_price) * base_sales

    # Total new demand
    new_sales = base_sales + own_demand_change + cross_demand_change

    # Profit calculation (assuming cost = 70% of original price)
    cost_per_unit = 0.7 * base_price

    original_profit = (base_price - cost_per_unit) * base_sales
    new_profit = (new_price - cost_per_unit) * new_sales

    profit_change_percent = ((new_profit - original_profit) / original_profit) * 100

    return profit_change_percent


# Example parameters based on our case study
base_params = {
    'product1': {'sales': 210, 'price': 54},
    'product2': {'sales': 200, 'price': 60}
}

elasticity_matrix = [
    [-2.3, 1.0],  # Product 1: own elasticity, cross elasticity
    [2.0, -4.0]  # Product 2: cross elasticity, own elasticity
]


def find_nash_equilibrium(elasticity_matrix, base_params):
    """
    Find Nash Equilibrium price changes using symbolic math

    Args:
        elasticity_matrix: 2x2 matrix of elasticities
        base_params: Dictionary with base sales and prices

    Returns:
        Optimal price changes for both products
    """
    # Extract parameters
    s1, p1 = base_params['product1']['sales'], base_params['product1']['price']
    s2, p2 = base_params['product2']['sales'], base_params['product2']['price']

    # Define symbolic variables for price changes
    x, y = symbols('x,y')  # x = product1 price change, y = product2 price change

    # Calculate coefficients for profit maximization equations
    # First-order conditions: ∂π/∂x = 0, ∂π/∂y = 0

    a1 = elasticity_matrix[0][0] * s1 + s1  # Own elasticity effect + base sales
    a2 = elasticity_matrix[1][1] * s2 + s2

    b1 = (elasticity_matrix[0][0] * 2 * s1) / p1  # Second-order own effect
    b2 = (elasticity_matrix[1][1] * 2 * s2) / p2

    c1 = (elasticity_matrix[0][1] * s1) / p2  # Cross effect on product 1
    c2 = (elasticity_matrix[1][0] * s2) / p1  # Cross effect on product 2

    # First-order conditions (profit maximization)
    eq1 = Eq(a1 + x * b1 + y * c1, 0)  # ∂π₁/∂x = 0
    eq2 = Eq(a2 + y * b2 + x * c2, 0)  # ∂π₂/∂y = 0

    # Solve system of equations
    solution = solve((eq1, eq2), (x, y))

    optimal_x = float(solution[x])
    optimal_y = float(solution[y])

    print(f"=== NASH EQUILIBRIUM ===")
    print(f"Product 1 optimal price change: ${optimal_x:.2f}")
    print(f"Product 2 optimal price change: ${optimal_y:.2f}")

    # Calculate resulting profits
    profit1 = calculate_profit_change(optimal_x, optimal_y,
                                      elasticity_matrix[0][0], elasticity_matrix[0][1],
                                      s1, p1)

    profit2 = calculate_profit_change(optimal_y, optimal_x,
                                      elasticity_matrix[1][1], elasticity_matrix[1][0],
                                      s2, p2)

    print(f"Product 1 profit change: {profit1:.1f}%")
    print(f"Product 2 profit change: {profit2:.1f}%")

    return optimal_x, optimal_y, profit1, profit2


# Find equilibrium
nash_x, nash_y, profit1, profit2 = find_nash_equilibrium(elasticity_matrix, base_params)


def visualize_competition_dynamics(elasticity_matrix, base_params, nash_x, nash_y):
    """
    Create visualizations showing competitive dynamics
    """
    s1, p1 = base_params['product1']['sales'], base_params['product1']['price']
    s2, p2 = base_params['product2']['sales'], base_params['product2']['price']

    # Create range of price changes for analysis
    price_changes = np.linspace(-30, 10, 100)

    # Fix competitor at Nash equilibrium, show best responses
    profits_1 = []
    profits_2 = []

    for price_change in price_changes:
        # Product 1 profits when product 2 is at Nash equilibrium
        profit_1 = calculate_profit_change(price_change, nash_y,
                                           elasticity_matrix[0][0], elasticity_matrix[0][1],
                                           s1, p1)
        profits_1.append(profit_1)

        # Product 2 profits when product 1 is at Nash equilibrium
        profit_2 = calculate_profit_change(price_change, nash_x,
                                           elasticity_matrix[1][1], elasticity_matrix[1][0],
                                           s2, p2)
        profits_2.append(profit_2)

    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Best response for Product 1
    ax1.plot(price_changes, profits_1, 'b-', linewidth=2, label='Product 1 Profit %')
    ax1.axvline(nash_x, color='red', linestyle='--', linewidth=2, label=f'Nash Equilibrium (${nash_x:.2f})')
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Product 1 Price Change ($)')
    ax1.set_ylabel('Profit Change (%)')
    ax1.set_title('Product 1 Best Response\n(Given Product 2 at Nash Equilibrium)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Best response for Product 2
    ax2.plot(price_changes, profits_2, 'g-', linewidth=2, label='Product 2 Profit %')
    ax2.axvline(nash_y, color='red', linestyle='--', linewidth=2, label=f'Nash Equilibrium (${nash_y:.2f})')
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Product 2 Price Change ($)')
    ax2.set_ylabel('Profit Change (%)')
    ax2.set_title('Product 2 Best Response\n(Given Product 1 at Nash Equilibrium)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Generate visualizations
visualize_competition_dynamics(elasticity_matrix, base_params, nash_x, nash_y)


def create_payoff_matrix_heatmap(elasticity_matrix, base_params):
    """
    Create heatmap showing payoff matrix for different strategy combinations
    """
    price_range = np.linspace(-25, 5, 10)
    payoff_matrix_1 = np.zeros((len(price_range), len(price_range)))
    payoff_matrix_2 = np.zeros((len(price_range), len(price_range)))

    s1, p1 = base_params['product1']['sales'], base_params['product1']['price']
    s2, p2 = base_params['product2']['sales'], base_params['product2']['price']

    for i, price_1 in enumerate(price_range):
        for j, price_2 in enumerate(price_range):
            payoff_matrix_1[i, j] = calculate_profit_change(price_1, price_2,
                                                            elasticity_matrix[0][0], elasticity_matrix[0][1],
                                                            s1, p1)
            payoff_matrix_2[i, j] = calculate_profit_change(price_2, price_1,
                                                            elasticity_matrix[1][1], elasticity_matrix[1][0],
                                                            s2, p2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Product 1 payoffs
    im1 = ax1.imshow(payoff_matrix_1, cmap='RdYlGn', aspect='auto')
    ax1.set_xlabel('Product 2 Price Change')
    ax1.set_ylabel('Product 1 Price Change')
    ax1.set_title('Product 1 Profit % Payoff Matrix')
    ax1.set_xticks(range(len(price_range)))
    ax1.set_xticklabels([f'${x:.1f}' for x in price_range])
    ax1.set_yticks(range(len(price_range)))
    ax1.set_yticklabels([f'${x:.1f}' for x in price_range])
    plt.colorbar(im1, ax=ax1)

    # Product 2 payoffs
    im2 = ax2.imshow(payoff_matrix_2, cmap='RdYlGn', aspect='auto')
    ax2.set_xlabel('Product 1 Price Change')
    ax2.set_ylabel('Product 2 Price Change')
    ax2.set_title('Product 2 Profit % Payoff Matrix')
    ax2.set_xticks(range(len(price_range)))
    ax2.set_xticklabels([f'${x:.1f}' for x in price_range])
    ax2.set_yticks(range(len(price_range)))
    ax2.set_yticklabels([f'${x:.1f}' for x in price_range])
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.show()


# Create payoff matrix visualization
create_payoff_matrix_heatmap(elasticity_matrix, base_params)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class GameTheoryPricer:
    """
    Production-ready game theory pricing system for e-commerce
    """

    def __init__(self, product_id, own_elasticity, cross_elasticity,
                 base_sales, base_price, cost_ratio=0.7):
        """
        Initialize the pricing system

        Args:
            product_id: Unique identifier for the product
            own_elasticity: Price elasticity of demand (negative value)
            cross_elasticity: Cross-price elasticity with main competitor
            base_sales: Current daily sales volume
            base_price: Current price
            cost_ratio: Cost as percentage of price (default 70%)
        """
        self.product_id = product_id
        self.own_elasticity = own_elasticity
        self.cross_elasticity = cross_elasticity
        self.base_sales = base_sales
        self.base_price = base_price
        self.cost_ratio = cost_ratio
        self.cost_per_unit = base_price * cost_ratio

        # Store historical data for learning
        self.price_history = []
        self.competitor_price_history = []
        self.sales_history = []
        self.profit_history = []

    def calculate_demand_change(self, own_price_change, competitor_price_change):
        """
        Calculate expected demand change based on elasticities
        """
        # Demand change due to own price change
        own_effect = self.own_elasticity * (own_price_change / self.base_price) * self.base_sales

        # Demand change due to competitor price change
        cross_effect = self.cross_elasticity * (competitor_price_change / self.base_price) * self.base_sales

        return own_effect + cross_effect

    def calculate_profit(self, price_change, competitor_price_change):
        """
        Calculate expected profit given price changes
        """
        new_price = self.base_price + price_change
        demand_change = self.calculate_demand_change(price_change, competitor_price_change)
        new_sales = max(0, self.base_sales + demand_change)  # Ensure non-negative sales

        profit = (new_price - self.cost_per_unit) * new_sales
        return profit, new_sales

    def predict_optimal_price_change(self, competitor_price_change):
        """
        Given competitor's price change, calculate optimal response using calculus
        """
        # For profit maximization: dπ/dp = 0
        # π = (p - c) * (q + elasticity * Δp/p_base * q_base + cross_elasticity * Δp_comp/p_base * q_base)

        # Optimal price change formula derived from first-order condition
        numerator = -(
                    self.base_sales + self.cross_elasticity * competitor_price_change * self.base_sales / self.base_price)
        denominator = 2 * self.own_elasticity * self.base_sales / self.base_price

        if denominator != 0:
            optimal_change = numerator / denominator
        else:
            optimal_change = 0

        return optimal_change

    def simulate_pricing_scenarios(self, competitor_price_changes, own_price_changes=None):
        """
        Evaluate multiple pricing scenarios and their outcomes
        """
        if own_price_changes is None:
            # Generate optimal responses to competitor changes
            own_price_changes = [self.predict_optimal_price_change(comp_change)
                                 for comp_change in competitor_price_changes]

        results = []
        for i, (own_change, comp_change) in enumerate(zip(own_price_changes, competitor_price_changes)):
            profit, sales = self.calculate_profit(own_change, comp_change)

            results.append({
                'scenario': i + 1,
                'own_price_change': own_change,
                'competitor_price_change': comp_change,
                'new_price': self.base_price + own_change,
                'expected_sales': sales,
                'expected_profit': profit,
                'profit_change_pct': ((profit - self.base_sales * (self.base_price - self.cost_per_unit)) /
                                      (self.base_sales * (self.base_price - self.cost_per_unit))) * 100
            })

        return pd.DataFrame(results)

    def update_market_data(self, current_price, competitor_price, actual_sales):
        """
        Update system with actual market data for learning
        """
        self.price_history.append(current_price)
        self.competitor_price_history.append(competitor_price)
        self.sales_history.append(actual_sales)

        current_profit = (current_price - self.cost_per_unit) * actual_sales
        self.profit_history.append(current_profit)

        # Update base parameters with recent data (simple moving average)
        if len(self.sales_history) >= 7:  # Weekly moving average
            self.base_sales = np.mean(self.sales_history[-7:])
            self.base_price = np.mean(self.price_history[-7:])

    def get_pricing_recommendation(self, competitor_price_change, max_change_pct=0.15):
        """
        Get pricing recommendation with business constraints
        """
        optimal_change = self.predict_optimal_price_change(competitor_price_change)

        # Apply business constraints
        max_change = self.base_price * max_change_pct
        constrained_change = np.clip(optimal_change, -max_change, max_change)

        profit, sales = self.calculate_profit(constrained_change, competitor_price_change)

        recommendation = {
            'recommended_price_change': constrained_change,
            'new_price': self.base_price + constrained_change,
            'expected_sales': sales,
            'expected_profit': profit,
            'confidence': 'High' if abs(constrained_change - optimal_change) < 0.01 else 'Medium'
        }

        return recommendation


class CompetitiveMarketSimulator:
    """
    Simulate a competitive market with multiple players using game theory
    """

    def __init__(self):
        self.players = {}
        self.market_history = []

    def add_player(self, player_id, pricer):
        """Add a player to the market simulation"""
        self.players[player_id] = pricer

    def simulate_market_day(self, external_shocks=None):
        """
        Simulate one day of market competition
        """
        if external_shocks is None:
            external_shocks = {}

        results = {}

        # Each player observes others and makes pricing decisions
        for player_id, pricer in self.players.items():
            # Get competitor's last price change (simplified for 2-player game)
            competitor_id = [pid for pid in self.players.keys() if pid != player_id][0]
            competitor_pricer = self.players[competitor_id]

            # Assume competitor maintains current strategy or random small change
            competitor_change = external_shocks.get(competitor_id, np.random.normal(0, 2))

            # Get pricing recommendation
            recommendation = pricer.get_pricing_recommendation(competitor_change)

            results[player_id] = {
                'price_change': recommendation['recommended_price_change'],
                'new_price': recommendation['new_price'],
                'expected_sales': recommendation['expected_sales'],
                'expected_profit': recommendation['expected_profit']
            }

        # Update all players with market outcomes
        for player_id, result in results.items():
            # Add some market noise to actual sales
            actual_sales = max(0, result['expected_sales'] + np.random.normal(0, 10))
            self.players[player_id].update_market_data(
                result['new_price'],
                results[[pid for pid in results.keys() if pid != player_id][0]]['new_price'],
                actual_sales
            )

        self.market_history.append({
            'date': datetime.now(),
            'results': results
        })

        return results


# === PRODUCTION DEMONSTRATION WITH SIMULATED DATA ===

def create_sample_ecommerce_scenario():
    """
    Create a realistic e-commerce pricing scenario
    """
    print("=== E-COMMERCE PRICING COMPETITION SIMULATION ===")
    print("Scenario: Two laptop sellers on competing platforms")
    print()

    # Create two competing products with realistic parameters
    laptop_amazon = GameTheoryPricer(
        product_id="LAPTOP_AMAZON",
        own_elasticity=-2.1,  # 10% price increase = 21% sales decrease
        cross_elasticity=0.8,  # Competitor 10% price increase = 8% our sales increase
        base_sales=150,  # Daily sales
        base_price=899,  # Current price
        cost_ratio=0.75  # 75% cost ratio
    )

    laptop_flipkart = GameTheoryPricer(
        product_id="LAPTOP_FLIPKART",
        own_elasticity=-1.9,
        cross_elasticity=0.7,
        base_sales=120,
        base_price=925,
        cost_ratio=0.78
    )

    # Demonstrate pricing recommendations
    print("1. PRICING RECOMMENDATIONS")
    print("-" * 40)

    # Scenario: Flipkart reduces price by $50
    flipkart_reduction = -50
    amazon_response = laptop_amazon.get_pricing_recommendation(flipkart_reduction)

    print(f"Competitor (Flipkart) reduces price by: ${abs(flipkart_reduction)}")
    print(f"Recommended Amazon response: ${amazon_response['recommended_price_change']:.2f}")
    print(f"New Amazon price: ${amazon_response['new_price']:.2f}")
    print(f"Expected sales increase: {amazon_response['expected_sales'] - laptop_amazon.base_sales:.1f} units/day")
    print(f"Confidence: {amazon_response['confidence']}")
    print()

    # Scenario analysis
    print("2. SCENARIO ANALYSIS")
    print("-" * 40)

    competitor_scenarios = [-75, -50, -25, 0, 25, 50]
    scenarios_df = laptop_amazon.simulate_pricing_scenarios(competitor_scenarios)

    print("Amazon's optimal responses to Flipkart price changes:")
    print(scenarios_df[['competitor_price_change', 'own_price_change', 'new_price',
                        'expected_sales', 'profit_change_pct']].round(2))
    print()

    # Market simulation
    print("3. MARKET SIMULATION (30 Days)")
    print("-" * 40)

    market = CompetitiveMarketSimulator()
    market.add_player("Amazon", laptop_amazon)
    market.add_player("Flipkart", laptop_flipkart)

    # Simulate 30 days of competition
    simulation_results = []

    for day in range(30):
        # Add some market events
        external_shocks = {}
        if day == 10:  # Black Friday-style event
            external_shocks["Flipkart"] = -30  # Aggressive price cut
        elif day == 20:  # Supply chain issues
            external_shocks["Amazon"] = 15  # Price increase

        daily_results = market.simulate_market_day(external_shocks)
        simulation_results.append(daily_results)

    # Analyze simulation results
    amazon_prices = []
    flipkart_prices = []
    amazon_profits = []
    flipkart_profits = []

    for day_result in simulation_results:
        amazon_prices.append(day_result["Amazon"]["new_price"])
        flipkart_prices.append(day_result["Flipkart"]["new_price"])
        amazon_profits.append(day_result["Amazon"]["expected_profit"])
        flipkart_profits.append(day_result["Flipkart"]["expected_profit"])

    # Display key metrics
    print(f"Final Amazon price: ${amazon_prices[-1]:.2f} (started at $899)")
    print(f"Final Flipkart price: ${flipkart_prices[-1]:.2f} (started at $925)")
    print(f"Amazon avg daily profit: ${np.mean(amazon_profits):.2f}")
    print(f"Flipkart avg daily profit: ${np.mean(flipkart_profits):.2f}")
    print()

    return market, simulation_results, scenarios_df


def visualize_market_simulation(simulation_results):
    """
    Create visualizations of the market simulation
    """
    days = range(1, len(simulation_results) + 1)

    amazon_prices = [day["Amazon"]["new_price"] for day in simulation_results]
    flipkart_prices = [day["Flipkart"]["new_price"] for day in simulation_results]
    amazon_profits = [day["Amazon"]["expected_profit"] for day in simulation_results]
    flipkart_profits = [day["Flipkart"]["expected_profit"] for day in simulation_results]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Price evolution
    ax1.plot(days, amazon_prices, 'b-', label='Amazon', linewidth=2)
    ax1.plot(days, flipkart_prices, 'r-', label='Flipkart', linewidth=2)
    ax1.axvline(10, color='gray', linestyle='--', alpha=0.7, label='Black Friday')
    ax1.axvline(20, color='gray', linestyle=':', alpha=0.7, label='Supply Issue')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Price Competition Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Profit evolution
    ax2.plot(days, amazon_profits, 'b-', label='Amazon', linewidth=2)
    ax2.plot(days, flipkart_profits, 'r-', label='Flipkart', linewidth=2)
    ax2.axvline(10, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(20, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Daily Profit ($)')
    ax2.set_title('Profit Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Price vs Profit scatter
    ax3.scatter(amazon_prices, amazon_profits, c='blue', alpha=0.6, label='Amazon')
    ax3.scatter(flipkart_prices, flipkart_profits, c='red', alpha=0.6, label='Flipkart')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Profit ($)')
    ax3.set_title('Price vs Profit Relationship')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Competitive dynamics
    price_diff = np.array(amazon_prices) - np.array(flipkart_prices)
    profit_diff = np.array(amazon_profits) - np.array(flipkart_profits)

    ax4.scatter(price_diff, profit_diff, c=days, cmap='viridis', alpha=0.7)
    ax4.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax4.axvline(0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Price Difference (Amazon - Flipkart)')
    ax4.set_ylabel('Profit Difference (Amazon - Flipkart)')
    ax4.set_title('Competitive Advantage Over Time')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Day')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demonstrate_real_time_pricing():
    """
    Demonstrate real-time pricing API for production use
    """
    print("4. REAL-TIME PRICING API DEMONSTRATION")
    print("-" * 50)

    # Initialize pricing engine
    pricing_engine = GameTheoryPricer(
        product_id="SMARTPHONE_X",
        own_elasticity=-1.8,
        cross_elasticity=0.6,
        base_sales=200,
        base_price=699,
        cost_ratio=0.72
    )

    # Simulate real-time competitor monitoring
    competitor_actions = [
        {"time": "09:00", "action": "price_drop", "amount": -20},
        {"time": "12:00", "action": "promotion", "amount": -35},
        {"time": "15:00", "action": "price_increase", "amount": 10},
        {"time": "18:00", "action": "flash_sale", "amount": -50}
    ]

    print("Real-time competitor monitoring and responses:")
    print("Time     | Competitor Action | Our Response     | Expected Impact")
    print("-" * 65)

    for action in competitor_actions:
        recommendation = pricing_engine.get_pricing_recommendation(action["amount"])

        impact = "Positive" if recommendation["expected_profit"] > pricing_engine.base_sales * (
                    pricing_engine.base_price - pricing_engine.cost_per_unit) else "Negative"

        print(
            f"{action['time']}    | {action['action']:15} | ${recommendation['recommended_price_change']:6.2f} change | {impact}")

        # Update with simulated actual results
        actual_sales = recommendation["expected_sales"] + np.random.normal(0, 5)
        pricing_engine.update_market_data(
            recommendation["new_price"],
            pricing_engine.base_price + action["amount"],
            actual_sales
        )


# === RUN THE COMPLETE DEMONSTRATION ===

if __name__ == "__main__":
    # Run the complete production demonstration
    market, simulation_results, scenarios_df = create_sample_ecommerce_scenario()

    # Show visualizations
    visualize_market_simulation(simulation_results)

    # Demonstrate real-time pricing
    demonstrate_real_time_pricing()

    print("\n=== PRODUCTION SYSTEM SUMMARY ===")
    print("This implementation provides:")
    print("• Real-time pricing recommendations based on competitor actions")
    print("• Multi-scenario analysis for strategic planning")
    print("• Market simulation for testing pricing strategies")
    print("• Historical data learning and adaptation")
    print("• Production-ready API for e-commerce integration")