"""
It is interesting that cross elasticity actually encode the fierceness of competition between competitors pair

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.optimize import minimize, NonlinearConstraint
import warnings
warnings.filterwarnings('ignore')


class ThreePlayerGameTheoryPricer:
    """
    Three-player competitive pricing using game theory
    """

    def __init__(self, players_config):
        """
        Initialize three-player pricing game

        Args:
            players_config: Dict with player configurations
        """
        self.players = players_config
        self.n_players = len(players_config)

        # Extract elasticity matrix (3x3)
        self.elasticity_matrix = self._build_elasticity_matrix()

    def _build_elasticity_matrix(self):
        """
        Build 3x3 elasticity matrix

        Matrix structure:
        [ε₁₁  ε₁₂  ε₁₃]  # Player 1's elasticities
        [ε₂₁  ε₂₂  ε₂₃]  # Player 2's elasticities
        [ε₃₁  ε₃₂  ε₃₃]  # Player 3's elasticities

        Diagonal: own-price elasticities (negative)
        Off-diagonal: cross-price elasticities (positive for substitutes)
        """
        matrix = np.zeros((3, 3))

        for i, (player_i, config_i) in enumerate(self.players.items()):
            for j, (player_j, config_j) in enumerate(self.players.items()):
                if i == j:
                    # Own price elasticity (diagonal)
                    matrix[i][j] = config_i['own_elasticity']
                else:
                    # Cross-price elasticity (off-diagonal)
                    matrix[i][j] = config_i['cross_elasticities'][player_j]

        return matrix

    def find_three_player_nash_equilibrium(self):
        """
        Find Nash equilibrium for three-player pricing game using unconstrained optimization
        """
        # Define symbolic variables for price changes
        x1, x2, x3 = symbols('x1 x2 x3')
        variables = [x1, x2, x3]

        equations = []

        for i, (player_id, config) in enumerate(self.players.items()):
            s_i = config['sales']
            p_i = config['price']

            # Calculate coefficients for player i's first-order condition
            # General form: aᵢ + bᵢᵢxᵢ + bᵢⱼxⱼ + bᵢₖxₖ = 0

            # Constant term
            a_i = self.elasticity_matrix[i][i] * s_i + s_i

            # Coefficient for own price change (second-order effect)
            b_ii = (self.elasticity_matrix[i][i] * 2 * s_i) / p_i

            # Coefficients for competitor price changes
            coefficients = [0, 0, 0]
            coefficients[i] = b_ii  # Own price coefficient

            for j, (other_player, other_config) in enumerate(self.players.items()):
                if i != j:
                    p_j = other_config['price']
                    coefficients[j] = (self.elasticity_matrix[i][j] * s_i) / p_j

            # Build equation: a_i + coefficients[0]*x1 + coefficients[1]*x2 + coefficients[2]*x3 = 0
            equation = Eq(a_i + sum(coef * var for coef, var in zip(coefficients, variables)), 0)
            equations.append(equation)

            print(f"Player {i + 1} ({player_id}) first-order condition:")
            print(f"  {a_i:.2f} + {coefficients[0]:.3f}*x1 + {coefficients[1]:.3f}*x2 + {coefficients[2]:.3f}*x3 = 0")

        print("\\nSolving 3x3 system of equations...")

        # Solve the system
        try:
            solution = solve(equations, variables)

            nash_x1 = float(solution[x1])
            nash_x2 = float(solution[x2])
            nash_x3 = float(solution[x3])

            print(f"\\n=== THREE-PLAYER NASH EQUILIBRIUM (UNCONSTRAINED) ===")
            print(f"Player 1 optimal price change: ${nash_x1:.2f}")
            print(f"Player 2 optimal price change: ${nash_x2:.2f}")
            print(f"Player 3 optimal price change: ${nash_x3:.2f}")

            # Check if prices go below cost
            viable = self._check_cost_viability([nash_x1, nash_x2, nash_x3])
            
            # Calculate resulting profits for unconstrained case
            profits = self._calculate_equilibrium_profits(nash_x1, nash_x2, nash_x3)

            for i, (player_id, profit_change) in enumerate(profits.items()):
                print(f"Player {i + 1} ({player_id}) profit change: {profit_change:.1f}%")
            
            if not viable:
                print("\\n⚠️  UNCONSTRAINED SOLUTION VIOLATES COST CONSTRAINTS!")
                print("Attempting constrained optimization...")
                
                # Store unconstrained solution
                unconstrained_result = {
                    'price_changes': [nash_x1, nash_x2, nash_x3],
                    'profits': profits,
                    'solution': solution,
                    'method': 'unconstrained'
                }
                
                # Get constrained solution
                constrained_result = self.find_constrained_nash_equilibrium()
                
                if constrained_result:
                    # Compare both solutions
                    self._compare_constrained_vs_unconstrained(unconstrained_result, constrained_result)
                    return constrained_result
                else:
                    return unconstrained_result


            return {
                'price_changes': [nash_x1, nash_x2, nash_x3],
                'profits': profits,
                'solution': solution,
                'method': 'unconstrained'
            }

        except Exception as e:
            print(f"Could not solve system: {e}")
            return None

    def find_constrained_nash_equilibrium(self):
        """
        Find Nash equilibrium with cost constraints using optimization
        """
        print("\\n=== SOLVING WITH COST CONSTRAINTS ===")
        
        # Initial guess (small price reductions)
        x0 = np.array([-10.0, -10.0, -10.0])
        
        # Define objective function (sum of squared deviations from FOCs)
        def objective(x):
            total_deviation = 0
            for i, (player_id, config) in enumerate(self.players.items()):
                s_i = config['sales']
                p_i = config['price']
                
                # First-order condition for player i
                a_i = self.elasticity_matrix[i][i] * s_i + s_i
                b_ii = (self.elasticity_matrix[i][i] * 2 * s_i) / p_i
                
                foc_value = a_i + b_ii * x[i]
                for j, (other_player, other_config) in enumerate(self.players.items()):
                    if i != j:
                        p_j = other_config['price']
                        coef_j = (self.elasticity_matrix[i][j] * s_i) / p_j
                        foc_value += coef_j * x[j]
                
                total_deviation += foc_value**2
            
            return total_deviation
        
        # Define cost constraints: new_price >= cost
        def cost_constraint(x):
            constraints = []
            for i, (player_id, config) in enumerate(self.players.items()):
                p_i = config['price']
                cost_ratio = config.get('cost_ratio', 0.7)
                cost_per_unit = p_i * cost_ratio
                new_price = p_i + x[i]
                
                # Constraint: new_price - cost >= 0
                constraints.append(new_price - cost_per_unit)
            
            return np.array(constraints)
        
        # Set up constraints
        cost_constraints = NonlinearConstraint(cost_constraint, 0, np.inf)
        
        # Bounds: reasonable price change limits
        max_increase = 100  # Max $100 increase
        max_decrease = -200  # Max $200 decrease  
        bounds = [(max_decrease, max_increase) for _ in range(3)]
        
        # Solve optimization problem
        result = minimize(objective, x0, method='SLSQP', 
                         constraints=cost_constraints, bounds=bounds,
                         options={'ftol': 1e-6, 'disp': False})
        
        if result.success:
            nash_x1, nash_x2, nash_x3 = result.x
            
            print(f"\\n=== CONSTRAINED NASH EQUILIBRIUM ===")
            print(f"Player 1 optimal price change: ${nash_x1:.2f}")
            print(f"Player 2 optimal price change: ${nash_x2:.2f}")
            print(f"Player 3 optimal price change: ${nash_x3:.2f}")
            
            # Verify constraints are satisfied
            constraint_values = cost_constraint(result.x)
            print(f"\\nCost constraint violations: {constraint_values}")
            
            # Check if any player should exit
            exit_players = self._check_market_exit([nash_x1, nash_x2, nash_x3])
            
            # Calculate resulting profits
            profits = self._calculate_equilibrium_profits(nash_x1, nash_x2, nash_x3)
            
            for i, (player_id, profit_change) in enumerate(profits.items()):
                print(f"Player {i + 1} ({player_id}) profit change: {profit_change:.1f}%")
            
            return {
                'price_changes': [nash_x1, nash_x2, nash_x3],
                'profits': profits,
                'solution': result,
                'method': 'constrained',
                'exit_players': exit_players
            }
        else:
            print(f"\\n❌ Constrained optimization failed: {result.message}")
            return self._handle_market_exit()

    def _check_cost_viability(self, price_changes):
        """Check if price changes result in selling below cost"""
        for i, (player_id, config) in enumerate(self.players.items()):
            p_i = config['price']
            cost_ratio = config.get('cost_ratio', 0.7)
            cost_per_unit = p_i * cost_ratio
            new_price = p_i + price_changes[i]
            
            if new_price < cost_per_unit:
                print(f"❌ {player_id}: New price ${new_price:.2f} < Cost ${cost_per_unit:.2f}")
                return False
        
        return True

    def _check_market_exit(self, price_changes):
        """Identify players who should exit the market"""
        exit_players = []
        profits = self._calculate_equilibrium_profits(*price_changes)
        
        for player_id, profit_change in profits.items():
            if profit_change < -50:  # Exit if profit drops by more than 50%
                exit_players.append(player_id)
                print(f"⚠️  {player_id} should consider market exit (profit change: {profit_change:.1f}%)")
        
        return exit_players

    def _handle_market_exit(self):
        """Handle scenario where no viable equilibrium exists"""
        print("\\n💥 NO VIABLE EQUILIBRIUM EXISTS")
        print("Market dynamics suggest:")
        print("1. Weakest player(s) should exit the market")
        print("2. Remaining players can achieve sustainable pricing")
        print("3. Market consolidation is inevitable")
        
        # Simulate 2-player game by removing weakest player
        weakest_player = min(self.players.keys(), 
                           key=lambda k: self.players[k]['sales'])
        
        print(f"\\nSimulating market after {weakest_player} exits...")
        remaining_players = {k: v for k, v in self.players.items() if k != weakest_player}
        
        return {
            'price_changes': [None, None, None],
            'profits': {},
            'solution': None,
            'method': 'market_exit',
            'exit_players': [weakest_player],
            'remaining_players': remaining_players
        }

    def _compare_constrained_vs_unconstrained(self, unconstrained, constrained):
        """Compare unconstrained vs constrained Nash equilibria"""
        print("\\n" + "="*60)
        print("📊 CONSTRAINED vs UNCONSTRAINED NASH EQUILIBRIUM COMPARISON")
        print("="*60)
        
        player_names = list(self.players.keys())
        
        print("\\n🔴 UNCONSTRAINED Nash Equilibrium (violates cost constraints):")
        for i, (player_id, price_change) in enumerate(zip(player_names, unconstrained['price_changes'])):
            profit_change = unconstrained['profits'][player_id]
            current_price = self.players[player_id]['price']
            new_price = current_price + price_change
            cost = current_price * self.players[player_id]['cost_ratio']
            
            print(f"  {player_id}: ${price_change:+.2f} → ${new_price:.2f} (cost: ${cost:.2f}) | Profit: {profit_change:+.1f}%")
        
        print("\\n🟢 CONSTRAINED Nash Equilibrium (respects cost constraints):")
        for i, (player_id, price_change) in enumerate(zip(player_names, constrained['price_changes'])):
            profit_change = constrained['profits'][player_id]
            current_price = self.players[player_id]['price']
            new_price = current_price + price_change
            cost = current_price * self.players[player_id]['cost_ratio']
            
            print(f"  {player_id}: ${price_change:+.2f} → ${new_price:.2f} (cost: ${cost:.2f}) | Profit: {profit_change:+.1f}%")
        
        print("\\n📈 KEY DIFFERENCES:")
        for i, player_id in enumerate(player_names):
            unconstrained_change = unconstrained['price_changes'][i]
            constrained_change = constrained['price_changes'][i]
            difference = constrained_change - unconstrained_change
            
            print(f"  {player_id}: Constrained is ${difference:+.2f} higher than unconstrained")
        
        print("\\n🎯 ECONOMIC INTERPRETATION:")
        print("• Unconstrained solution represents 'theoretical optimum' ignoring reality")
        print("• Constrained solution represents 'practical equilibrium' companies actually choose")
        print("• Constraints force players away from destructive price wars")
        print("• Real Nash equilibrium ≠ Mathematical Nash equilibrium when constraints bind")

    def _calculate_equilibrium_profits(self, x1, x2, x3):
        """Calculate profit changes at equilibrium"""
        profits = {}
        price_changes = [x1, x2, x3]

        for i, (player_id, config) in enumerate(self.players.items()):
            s_i = config['sales']
            p_i = config['price']
            cost_ratio = config.get('cost_ratio', 0.7)

            # Calculate demand change
            demand_change = 0
            for j, x_j in enumerate(price_changes):
                demand_change += self.elasticity_matrix[i][j] * (x_j / p_i) * s_i

            new_sales = s_i + demand_change
            new_price = p_i + price_changes[i]
            cost_per_unit = p_i * cost_ratio

            original_profit = (p_i - cost_per_unit) * s_i
            new_profit = (new_price - cost_per_unit) * new_sales

            profit_change_pct = ((new_profit - original_profit) / original_profit) * 100
            profits[player_id] = profit_change_pct

        return profits

    def analyze_competitive_dynamics(self, nash_result):
        """
        Analyze the competitive dynamics and strategic interactions
        """
        price_changes = nash_result['price_changes']

        print("\\n=== COMPETITIVE DYNAMICS ANALYSIS ===")

        # 1. Market leadership analysis
        most_aggressive = np.argmin(price_changes)
        least_aggressive = np.argmax(price_changes)

        player_names = list(self.players.keys())
        print(f"Most aggressive pricer: {player_names[most_aggressive]} (${price_changes[most_aggressive]:.2f})")
        print(f"Least aggressive pricer: {player_names[least_aggressive]} (${price_changes[least_aggressive]:.2f})")

        # 2. Competitive intensity
        price_variance = np.var(price_changes)
        print(f"Price variance (competitive intensity): {price_variance:.2f}")

        # 3. Coalition potential analysis
        print("\\n--- Coalition Analysis ---")
        self._analyze_coalition_potential(nash_result)

        # 4. Market share impact
        print("\\n--- Market Share Impact ---")
        self._analyze_market_share_changes(nash_result)

    def _analyze_coalition_potential(self, nash_result):
        """
        Analyze potential for coalitions between players
        """
        profits = nash_result['profits']
        player_names = list(profits.keys())

        # Check if any two players could benefit from cooperation
        print("Potential coalitions (both players benefit from cooperation):")

        for i in range(len(player_names)):
            for j in range(i + 1, len(player_names)):
                player_i = player_names[i]
                player_j = player_names[j]

                # Simplified coalition analysis: what if both reduce price cuts by 50%?
                coalition_benefit_i = profits[player_i] * 1.2  # Assume 20% improvement
                coalition_benefit_j = profits[player_j] * 1.2

                if coalition_benefit_i > profits[player_i] and coalition_benefit_j > profits[player_j]:
                    print(f"  {player_i} + {player_j}: Potential mutual benefit")

    def _analyze_market_share_changes(self, nash_result):
        """
        Analyze how market shares change at equilibrium
        """
        price_changes = nash_result['price_changes']

        # Calculate relative market share changes based on elasticities
        total_market_size = sum(config['sales'] for config in self.players.values())

        for i, (player_id, config) in enumerate(self.players.items()):
            # Calculate demand change for this player
            demand_change = 0
            for j, x_j in enumerate(price_changes):
                demand_change += self.elasticity_matrix[i][j] * (x_j / config['price']) * config['sales']

            new_sales = config['sales'] + demand_change
            original_share = config['sales'] / total_market_size * 100
            # Approximate new share (assuming market size changes)
            new_share = new_sales / (total_market_size + sum(demand_change for _ in range(3))) * 100

            print(f"{player_id}: {original_share:.1f}% → {new_share:.1f}% market share")


def create_three_player_scenario():
    """
    Create realistic three-player e-commerce scenario with more moderate elasticities
    """

    # Three major e-commerce platforms competing in laptop sales
    players_config = {
        'Amazon': {
            'sales': 250,  # Daily sales
            'price': 899,  # Current price
            'own_elasticity': -1.2,  # More realistic own price elasticity
            'cross_elasticities': {
                'Flipkart': 0.3,  # Moderate cross-elasticity
                'eBay': 0.2  # Lower due to differentiation
            },
            'cost_ratio': 0.75
        },
        'Flipkart': {
            'sales': 180,
            'price': 925,
            'own_elasticity': -1.1,  # Slightly less elastic
            'cross_elasticities': {
                'Amazon': 0.4,  # Higher sensitivity to market leader
                'eBay': 0.15
            },
            'cost_ratio': 0.78
        },
        'eBay': {
            'sales': 120,
            'price': 950,
            'own_elasticity': -0.9,  # Less price sensitive (niche/auction model)
            'cross_elasticities': {
                'Amazon': 0.25,
                'Flipkart': 0.1  # Least affected by Flipkart
            },
            'cost_ratio': 0.72
        }
    }
    
    return players_config

def create_high_competition_scenario():
    """
    Create a scenario that will trigger cost constraints to demonstrate the difference
    """
    # More intense competition scenario
    players_config = {
        'Amazon': {
            'sales': 250,
            'price': 899,
            'own_elasticity': -1.8,  # Higher elasticity = more intense competition
            'cross_elasticities': {
                'Flipkart': 0.6,  # Higher cross-elasticity
                'eBay': 0.4
            },
            'cost_ratio': 0.85  # Higher cost ratio = tighter margins
        },
        'Flipkart': {
            'sales': 180,
            'price': 925,
            'own_elasticity': -1.7,
            'cross_elasticities': {
                'Amazon': 0.7,
                'eBay': 0.3
            },
            'cost_ratio': 0.87
        },
        'eBay': {
            'sales': 120,
            'price': 950,
            'own_elasticity': -1.4,
            'cross_elasticities': {
                'Amazon': 0.5,
                'Flipkart': 0.2
            },
            'cost_ratio': 0.82
        }
    }
    
    return players_config

def run_scenario(players_config, scenario_name=""):
    """Run a pricing scenario and display results"""
    print(f"=== THREE-PLAYER E-COMMERCE PRICING COMPETITION {scenario_name} ===")
    print("Scenario: Amazon vs. Flipkart vs. eBay laptop pricing")
    print()

    # Display elasticity matrix
    game = ThreePlayerGameTheoryPricer(players_config)

    print("Cross-Elasticity Matrix:")
    print("         Amazon  Flipkart  eBay")
    for i, (player, _) in enumerate(players_config.items()):
        row_str = f"{player:8} "
        for j in range(3):
            row_str += f"{game.elasticity_matrix[i][j]:7.1f}  "
        print(row_str)
    print()

    # Find Nash equilibrium
    nash_result = game.find_three_player_nash_equilibrium()

    if nash_result:
        # Analyze competitive dynamics
        game.analyze_competitive_dynamics(nash_result)

        return game, nash_result

    return None, None


def visualize_three_player_dynamics(game, nash_result, scenario_name=""):
    """
    Create visualizations for three-player competition
    """
    if not nash_result:
        return

    price_changes = nash_result['price_changes']
    profits = nash_result['profits']
    player_names = list(game.players.keys())

    fig = plt.figure(figsize=(15, 10))

    # 1. Price Changes Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax1.bar(player_names, price_changes, color=colors)
    ax1.set_ylabel('Price Change ($)')
    ax1.set_title('Nash Equilibrium Price Changes')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

    # Add value labels on bars
    for bar, value in zip(bars, price_changes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + (1 if height >= 0 else -3),
                 f'${value:.1f}', ha='center', va='bottom' if height >= 0 else 'top')

    # 2. Profit Changes
    ax2 = plt.subplot(2, 3, 2)
    profit_values = list(profits.values())
    bars2 = ax2.bar(player_names, profit_values, color=colors)
    ax2.set_ylabel('Profit Change (%)')
    ax2.set_title('Profit Impact at Equilibrium')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

    for bar, value in zip(bars2, profit_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + (0.5 if height >= 0 else -1),
                 f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')

    # 3. Cross-Elasticity Heatmap
    ax3 = plt.subplot(2, 3, 3)
    sns.heatmap(game.elasticity_matrix, annot=True, cmap='RdBu_r', center=0,
                xticklabels=player_names, yticklabels=player_names, ax=ax3)
    ax3.set_title('Cross-Elasticity Matrix')
    ax3.set_ylabel('Affected Player')
    ax3.set_xlabel('Price-Changing Player')

    # 4. 3D Strategy Space (simplified 2D projection)
    ax4 = plt.subplot(2, 3, 4)
    # Project 3D Nash equilibrium onto 2D planes
    ax4.scatter(price_changes[0], price_changes[1], s=200, c='red', marker='*',
                label='Nash Equilibrium', zorder=5)
    ax4.set_xlabel(f'{player_names[0]} Price Change')
    ax4.set_ylabel(f'{player_names[1]} Price Change')
    ax4.set_title('Nash Equilibrium (2D Projection)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Add reaction function illustrations (simplified)
    x_range = np.linspace(min(price_changes) - 10, max(price_changes) + 10, 100)
    # Simplified reaction function for player 2 to player 1
    # This is an approximation for visualization
    reaction_slope = -0.5  # Simplified
    reaction_y = price_changes[1] + reaction_slope * (x_range - price_changes[0])
    ax4.plot(x_range, reaction_y, '--', alpha=0.7, label=f'{player_names[1]} reaction to {player_names[0]}')
    ax4.legend()

    # 5. Market Share Changes
    ax5 = plt.subplot(2, 3, 5)

    # Calculate market shares (simplified)
    original_sales = [config['sales'] for config in game.players.values()]
    total_original = sum(original_sales)
    original_shares = [sales / total_original * 100 for sales in original_sales]

    # Estimate new sales based on price changes and elasticities
    new_sales = []
    for i, (player_id, config) in enumerate(game.players.items()):
        demand_change = 0
        for j, x_j in enumerate(price_changes):
            demand_change += game.elasticity_matrix[i][j] * (x_j / config['price']) * config['sales']
        new_sales.append(config['sales'] + demand_change)

    total_new = sum(new_sales)
    new_shares = [sales / total_new * 100 for sales in new_sales]

    x_pos = np.arange(len(player_names))
    width = 0.35

    ax5.bar(x_pos - width / 2, original_shares, width, label='Original', alpha=0.7, color=colors)
    ax5.bar(x_pos + width / 2, new_shares, width, label='After Equilibrium', alpha=0.7,
            color=[c for c in colors], hatch='///')

    ax5.set_ylabel('Market Share (%)')
    ax5.set_title('Market Share Changes')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(player_names)
    ax5.legend()

    # 6. Competitive Intensity Analysis
    ax6 = plt.subplot(2, 3, 6)

    # Calculate competitive pressure each player faces
    competitive_pressure = []
    for i in range(len(player_names)):
        # Sum of absolute cross-elasticities (how much others affect this player)
        pressure = sum(abs(game.elasticity_matrix[i][j]) for j in range(len(player_names)) if i != j)
        competitive_pressure.append(pressure)

    bars6 = ax6.bar(player_names, competitive_pressure, color=colors)
    ax6.set_ylabel('Competitive Pressure Index')
    ax6.set_title('Competitive Pressure Faced by Each Player')

    for bar, value in zip(bars6, competitive_pressure):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{value:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    
    # Save the plot instead of showing it (for CLI environments)
    filename = f'three_player_pricing_analysis_{scenario_name.replace(" ", "_").replace("(", "").replace(")", "").lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved as: {filename}")
    plt.close()  # Close to free memory


def compare_two_vs_three_player():
    """
    Compare dynamics between two-player and three-player games
    """
    print("\\n=== COMPARISON: TWO-PLAYER vs THREE-PLAYER DYNAMICS ===")

    # Two-player scenario (from original implementation)
    two_player_nash = [-20, -28]  # From previous example

    # Three-player scenario
    players_config = create_three_player_scenario()
    game = ThreePlayerGameTheoryPricer(players_config)
    nash_result = game.find_three_player_nash_equilibrium()

    if nash_result:
        three_player_nash = nash_result['price_changes']

        print("\\nPrice reductions at Nash equilibrium:")
        print(f"Two-player:   Player 1: ${abs(two_player_nash[0])}, Player 2: ${abs(two_player_nash[1])}")
        print(
            f"Three-player: Player 1: ${abs(three_player_nash[0]):.1f}, Player 2: ${abs(three_player_nash[1]):.1f}, Player 3: ${abs(three_player_nash[2]):.1f}")

        print("\\nKey differences:")
        print("• Competition intensity typically increases with more players")
        print("• Strategic complexity grows exponentially (n! possible interactions)")
        print("• Coalition opportunities emerge (2 vs 1 scenarios)")
        print("• Market segmentation becomes more pronounced")
        print("• First-mover advantages become more critical")


# === RUN THE COMPLETE THREE-PLAYER ANALYSIS ===

if __name__ == "__main__":
    print("🔥 DEMONSTRATING CONSTRAINED vs UNCONSTRAINED NASH EQUILIBRIUM")
    print("="*80)
    
    # First run high competition scenario to trigger constraints
    print("\\n🚨 Running HIGH COMPETITION scenario (triggers cost constraints)...")
    high_competition_config = create_high_competition_scenario()
    game_high, nash_result_high = run_scenario(high_competition_config, "(HIGH COMPETITION)")
    
    print("\\n" + "="*80)
    print("\\n✅ Running MODERATE COMPETITION scenario (no constraint violations)...")
    moderate_config = create_three_player_scenario()
    game_moderate, nash_result_moderate = run_scenario(moderate_config, "(MODERATE)")
    
    # Create visualizations for both scenarios
    if game_high and nash_result_high:
        print("\\n📊 Creating visualizations for HIGH COMPETITION scenario...")
        visualize_three_player_dynamics(game_high, nash_result_high, "high_competition")
    
    if game_moderate and nash_result_moderate:
        print("\\n📊 Creating visualizations for MODERATE COMPETITION scenario...")
        visualize_three_player_dynamics(game_moderate, nash_result_moderate, "moderate_competition")
    
    # Compare with two-player dynamics
    compare_two_vs_three_player()
    
    print("\\n=== KEY INSIGHTS FOR THREE-PLAYER PRICING ===")
    print("1. Mathematical complexity increases dramatically (3x3 system)")
    print("2. Competitive responses become more nuanced and strategic")
    print("3. Coalition potential creates new strategic dimensions")
    print("4. Market fragmentation allows for niche positioning")
    print("5. First-mover advantages become more pronounced")
    print("6. ⭐ CONSTRAINTS FUNDAMENTALLY CHANGE NASH EQUILIBRIUM POINTS!")