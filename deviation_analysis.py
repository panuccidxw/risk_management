# Nash Equilibrium Deviation Analysis

"""
Nash Equilibrium Deviation Analysis
The Question: What Happens When Players Deviate?
Nash Equilibrium: Player 1 = -$20, Player 2 = -$28
If one party deviates from equilibrium, can the other take advantage by deviating to increase profit?
The answer is YES - but with important nuances. Let me demonstrate mathematically.
Mathematical Framework
Using our e-commerce example with:

Player 1: Base price $54, sales 210 units, elasticities [-2.3, 1.0]
Player 2: Base price $60, sales 200 units, elasticities [2.0, -4.0]

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_profit_change(price_change_1, price_change_2, player=1):
    """
    Calculate profit change for a player given both price changes
    """
    # Player 1 parameters
    s1, p1 = 210, 54
    elasticity_11, elasticity_12 = -2.3, 1.0
    cost_ratio_1 = 0.7

    # Player 2 parameters
    s2, p2 = 200, 60
    elasticity_21, elasticity_22 = 2.0, -4.0
    cost_ratio_2 = 0.7

    if player == 1:
        # Player 1's profit calculation
        s, p = s1, p1
        own_elasticity, cross_elasticity = elasticity_11, elasticity_12
        competitor_price_change = price_change_2
        competitor_base_price = p2
        own_price_change = price_change_1
    else:
        # Player 2's profit calculation
        s, p = s2, p2
        own_elasticity, cross_elasticity = elasticity_22, elasticity_21
        competitor_price_change = price_change_1
        competitor_base_price = p1
        own_price_change = price_change_2

    # Calculate demand changes
    own_demand_change = own_elasticity * (own_price_change / p) * s
    cross_demand_change = cross_elasticity * (competitor_price_change / competitor_base_price) * s

    new_sales = s + own_demand_change + cross_demand_change
    new_price = p + own_price_change
    cost_per_unit = p * (0.7 if player == 1 else 0.7)

    # Profit calculation
    original_profit = (p - cost_per_unit) * s
    new_profit = (new_price - cost_per_unit) * new_sales

    profit_change_pct = ((new_profit - original_profit) / original_profit) * 100

    return profit_change_pct, new_profit, new_sales


def analyze_deviation_scenarios():
    """
    Analyze what happens when players deviate from Nash equilibrium
    """
    nash_eq = [-20, -28]  # Nash equilibrium

    print("=== NASH EQUILIBRIUM DEVIATION ANALYSIS ===")
    print(f"Nash Equilibrium: Player 1 = ${nash_eq[0]}, Player 2 = ${nash_eq[1]}")
    print()

    # Calculate profits at Nash equilibrium
    nash_profit_1, _, _ = calculate_profit_change(nash_eq[0], nash_eq[1], player=1)
    nash_profit_2, _, _ = calculate_profit_change(nash_eq[0], nash_eq[1], player=2)

    print(f"Profits at Nash Equilibrium:")
    print(f"Player 1: {nash_profit_1:.2f}% change")
    print(f"Player 2: {nash_profit_2:.2f}% change")
    print()

    # Scenario 1: Player 1 deviates, Player 2 stays at Nash
    print("SCENARIO 1: Player 1 Deviates, Player 2 Stays at Nash")
    print("-" * 55)

    player1_deviations = [-30, -25, -15, -10, -5, 0, 5]
    player2_nash = nash_eq[1]  # Player 2 stays at -28

    results_1 = []
    for deviation in player1_deviations:
        profit_1, _, _ = calculate_profit_change(deviation, player2_nash, player=1)
        profit_2, _, _ = calculate_profit_change(deviation, player2_nash, player=2)

        results_1.append({
            'Player1_Price_Change': deviation,
            'Player2_Price_Change': player2_nash,
            'Player1_Profit_Change': profit_1,
            'Player2_Profit_Change': profit_2,
            'Player1_vs_Nash': profit_1 - nash_profit_1,
            'Player2_vs_Nash': profit_2 - nash_profit_2
        })

    df1 = pd.DataFrame(results_1)
    print(df1[['Player1_Price_Change', 'Player1_Profit_Change', 'Player1_vs_Nash',
               'Player2_Profit_Change', 'Player2_vs_Nash']].round(2))
    print()

    # Find Player 1's best deviation
    best_dev_1 = df1.loc[df1['Player1_Profit_Change'].idxmax()]
    print(f"Player 1's best deviation when Player 2 stays at Nash:")
    print(f"  Deviation: ${best_dev_1['Player1_Price_Change']}")
    print(f"  Profit improvement: {best_dev_1['Player1_vs_Nash']:.2f} percentage points")
    print(f"  But Player 2 also benefits: {best_dev_1['Player2_vs_Nash']:.2f} percentage points")
    print()

    # Scenario 2: Player 2 deviates, Player 1 stays at Nash
    print("SCENARIO 2: Player 2 Deviates, Player 1 Stays at Nash")
    print("-" * 55)

    player2_deviations = [-40, -35, -30, -25, -20, -15, -10]
    player1_nash = nash_eq[0]  # Player 1 stays at -20

    results_2 = []
    for deviation in player2_deviations:
        profit_1, _, _ = calculate_profit_change(player1_nash, deviation, player=1)
        profit_2, _, _ = calculate_profit_change(player1_nash, deviation, player=2)

        results_2.append({
            'Player1_Price_Change': player1_nash,
            'Player2_Price_Change': deviation,
            'Player1_Profit_Change': profit_1,
            'Player2_Profit_Change': profit_2,
            'Player1_vs_Nash': profit_1 - nash_profit_1,
            'Player2_vs_Nash': profit_2 - nash_profit_2
        })

    df2 = pd.DataFrame(results_2)
    print(df2[['Player2_Price_Change', 'Player2_Profit_Change', 'Player2_vs_Nash',
               'Player1_Profit_Change', 'Player1_vs_Nash']].round(2))
    print()

    # Find Player 2's best deviation
    best_dev_2 = df2.loc[df2['Player2_Profit_Change'].idxmax()]
    print(f"Player 2's best deviation when Player 1 stays at Nash:")
    print(f"  Deviation: ${best_dev_2['Player2_Price_Change']}")
    print(f"  Profit improvement: {best_dev_2['Player2_vs_Nash']:.2f} percentage points")
    print(f"  But Player 1 also benefits: {best_dev_2['Player1_vs_Nash']:.2f} percentage points")
    print()

    return df1, df2, nash_eq


def calculate_optimal_response_to_deviation(deviating_player, deviation_price, responding_player,
                                            search_range=None, precision=0.1):
    """
    Calculate the optimal response when one player deviates from Nash equilibrium

    Args:
        deviating_player: 1 or 2 (which player is deviating)
        deviation_price: The price change the deviating player chooses
        responding_player: 1 or 2 (which player is responding optimally)
        search_range: Tuple (min, max) for response search range
        precision: Step size for optimization search

    Returns:
        Dictionary with optimal response analysis
    """
    nash_eq = [-20, -28]

    if search_range is None:
        search_range = (-50, 10)  # Default search range

    response_prices = np.arange(search_range[0], search_range[1], precision)

    best_response_profit = -float('inf')
    best_response_price = None
    best_response_sales = None
    response_analysis = []

    for response_price in response_prices:
        if deviating_player == 1:
            # Player 1 deviates, Player 2 responds
            player1_price = deviation_price
            player2_price = response_price
        else:
            # Player 2 deviates, Player 1 responds
            player1_price = response_price
            player2_price = deviation_price

        profit_change, profit_absolute, sales = calculate_profit_change(
            player1_price, player2_price, player=responding_player
        )

        response_analysis.append({
            'response_price': response_price,
            'profit_change_pct': profit_change,
            'profit_absolute': profit_absolute,
            'sales': sales
        })

        if profit_change > best_response_profit:
            best_response_profit = profit_change
            best_response_price = response_price
            best_response_sales = sales

    # Calculate profits at Nash for comparison
    nash_profit, nash_profit_abs, nash_sales = calculate_profit_change(
        nash_eq[0], nash_eq[1], player=responding_player
    )

    # Calculate what happens to the deviating player
    if deviating_player == 1:
        deviator_profit_at_nash, _, _ = calculate_profit_change(nash_eq[0], nash_eq[1], player=1)
        deviator_profit_with_response, _, _ = calculate_profit_change(
            deviation_price, best_response_price, player=1)
    else:
        deviator_profit_at_nash, _, _ = calculate_profit_change(nash_eq[0], nash_eq[1], player=2)
        deviator_profit_with_response, _, _ = calculate_profit_change(
            best_response_price, deviation_price, player=2)

    return {
        'deviating_player': deviating_player,
        'deviation_price': deviation_price,
        'responding_player': responding_player,
        'optimal_response_price': best_response_price,
        'responder_profit_at_nash': nash_profit,
        'responder_optimal_profit': best_response_profit,
        'responder_profit_improvement': best_response_profit - nash_profit,
        'responder_sales_change': best_response_sales - nash_sales,
        'deviator_profit_at_nash': deviator_profit_at_nash,
        'deviator_profit_after_response': deviator_profit_with_response,
        'deviator_punishment': deviator_profit_with_response - deviator_profit_at_nash,
        'response_analysis': response_analysis}


def analyze_systematic_deviations():
    """
    Systematically analyze optimal responses to various deviations from Nash
    """
    nash_eq = [-20, -28]

    print("=== SYSTEMATIC DEVIATION ANALYSIS ===")
    print("Calculating optimal responses to various deviations from Nash equilibrium")
    print()

    # Player 1 deviations (from Nash -20)
    player1_deviations = [-30, -25, -15, -10, -5, 0, 5]

    print("PLAYER 1 DEVIATIONS (Player 2 responds optimally)")
    print("-" * 70)
    print("P1 Deviation | P2 Optimal Response | P2 Profit Gain | P1 Punishment")
    print("-" * 70)

    for deviation in player1_deviations:
        if deviation == nash_eq[0]:  # Skip Nash point
            continue

        response = calculate_optimal_response_to_deviation(
            deviating_player=1,
            deviation_price=deviation,
            responding_player=2
        )

        print(
            f"    ${deviation:6.0f}   |      ${response['optimal_response_price']:6.1f}       |    {response['responder_profit_improvement']:6.2f}%    |    {response['deviator_punishment']:6.2f}%")

    print()

    # Player 2 deviations (from Nash -28)
    player2_deviations = [-40, -35, -30, -25, -20, -15, -10]

    print("PLAYER 2 DEVIATIONS (Player 1 responds optimally)")
    print("-" * 70)
    print("P2 Deviation | P1 Optimal Response | P1 Profit Gain | P2 Punishment")
    print("-" * 70)

    for deviation in player2_deviations:
        if deviation == nash_eq[1]:  # Skip Nash point
            continue

        response = calculate_optimal_response_to_deviation(
            deviating_player=2,
            deviation_price=deviation,
            responding_player=1
        )

        print(
            f"    ${deviation:6.0f}   |      ${response['optimal_response_price']:6.1f}       |    {response['responder_profit_improvement']:6.2f}%    |    {response['deviator_punishment']:6.2f}%")

    return


def demonstrate_best_response_function():
    """
    Demonstrate the best response function concept with specific examples
    """
    print("\n=== BEST RESPONSE FUNCTION DEMONSTRATION ===")
    print()

    # Example 1: Player 1 deviates to -15 (less aggressive)
    print("EXAMPLE 1: Player 1 becomes less aggressive")
    print("-" * 50)
    deviation_1 = -15
    response_1 = calculate_optimal_response_to_deviation(
        deviating_player=1,
        deviation_price=deviation_1,
        responding_player=2,
        precision=0.5
    )

    print(f"Player 1 deviates from Nash (-20) to {deviation_1}")
    print(f"Player 2's optimal response: {response_1['optimal_response_price']}")
    print(f"Player 2's profit improvement: {response_1['responder_profit_improvement']:.2f} percentage points")
    print(f"Player 1's punishment: {response_1['deviator_punishment']:.2f} percentage points")
    print()

    print("Interpretation:")
    print(f"• Player 1 reduces price cut from $20 to $15 (becomes less competitive)")
    print(f"• Player 2's best response: reduce their price cut to ${abs(response_1['optimal_response_price'])}")
    print(f"• Player 2 benefits significantly from reduced competition")
    print(f"• Player 1 gets punished for the deviation")
    print()

    # Example 2: Player 1 deviates to -25 (more aggressive)
    print("EXAMPLE 2: Player 1 becomes more aggressive")
    print("-" * 50)
    deviation_2 = -25
    response_2 = calculate_optimal_response_to_deviation(
        deviating_player=1,
        deviation_price=deviation_2,
        responding_player=2,
        precision=0.5
    )

    print(f"Player 1 deviates from Nash (-20) to {deviation_2}")
    print(f"Player 2's optimal response: {response_2['optimal_response_price']}")
    print(f"Player 2's profit change: {response_2['responder_profit_improvement']:.2f} percentage points")
    print(f"Player 1's result: {response_2['deviator_punishment']:.2f} percentage points")
    print()

    print("Interpretation:")
    print(f"• Player 1 increases price cut from $20 to $25 (becomes more aggressive)")
    print(f"• Player 2's best response: increase their price cut to ${abs(response_2['optimal_response_price'])}")
    print(f"• Competition intensifies, both players may suffer")
    print(f"• This shows why price wars can be destructive")
    print()

    # Example 3: Demonstrate the range of responses
    print("EXAMPLE 3: Response function across deviation range")
    print("-" * 50)

    deviations = [-30, -25, -20, -15, -10, -5, 0]
    responses = []

    for dev in deviations:
        if dev == -20:  # Nash point
            responses.append(-28)  # Nash response
        else:
            resp = calculate_optimal_response_to_deviation(1, dev, 2, precision=1.0)
            responses.append(resp['optimal_response_price'])

    print("Player 1 Action | Player 2 Best Response | Interpretation")
    print("-" * 65)
    for dev, resp in zip(deviations, responses):
        if dev == -20:
            interpretation = "Nash Equilibrium"
        elif dev > -20:
            interpretation = "P1 less aggressive → P2 reduces cuts"
        else:
            interpretation = "P1 more aggressive → P2 increases cuts"

        print(f"     ${dev:6.0f}    |        ${resp:6.1f}        | {interpretation}")


def visualize_response_functions():
    """
    Create visualization of best response functions
    """
    print("\n=== RESPONSE FUNCTION VISUALIZATION ===")

    # Generate Player 1's best response to Player 2's actions
    p2_actions = np.arange(-40, -10, 1)
    p1_best_responses = []

    for p2_action in p2_actions:
        resp = calculate_optimal_response_to_deviation(2, p2_action, 1, precision=0.5)
        p1_best_responses.append(resp['optimal_response_price'])

    # Generate Player 2's best response to Player 1's actions
    p1_actions = np.arange(-35, -5, 1)
    p2_best_responses = []

    for p1_action in p1_actions:
        resp = calculate_optimal_response_to_deviation(1, p1_action, 2, precision=0.5)
        p2_best_responses.append(resp['optimal_response_price'])

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot best response functions
    plt.plot(p2_actions, p1_best_responses, 'b-', linewidth=2,
             label='Player 1 Best Response to Player 2')
    plt.plot(p2_best_responses, p1_actions, 'r-', linewidth=2,
             label='Player 2 Best Response to Player 1')

    # Mark Nash equilibrium
    nash_eq = [-20, -28]
    plt.scatter([nash_eq[1]], [nash_eq[0]], color='black', s=150,
                marker='*', zorder=5, label='Nash Equilibrium')

    # Mark some example deviations and responses
    examples = [
        (-15, calculate_optimal_response_to_deviation(1, -15, 2)['optimal_response_price'], 'P1 deviates to -15'),
        (-25, calculate_optimal_response_to_deviation(1, -25, 2)['optimal_response_price'], 'P1 deviates to -25')
    ]

    for p1_dev, p2_resp, label in examples:
        plt.scatter([p2_resp], [p1_dev], s=100, marker='o', zorder=4)
        plt.annotate(label, (p2_resp, p1_dev), xytext=(5, 5),
                     textcoords='offset points', fontsize=9)
        # Draw arrow from Nash to deviation
        plt.arrow(nash_eq[1], nash_eq[0], p2_resp - nash_eq[1], p1_dev - nash_eq[0],
                  head_width=0.5, head_length=0.3, fc='gray', ec='gray', alpha=0.6)

    plt.xlabel('Player 2 Price Change ($)')
    plt.ylabel('Player 1 Price Change ($)')
    plt.title('Best Response Functions and Nash Equilibrium')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add annotations
    plt.text(-35, -10, 'Nash Equilibrium:\nNo player can improve\nby unilateral deviation',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.tight_layout()
    plt.show()


def analyze_optimal_responses():
    """
    Enhanced optimal response analysis using the new function
    """
    print("SCENARIO 3: Detailed Optimal Response Analysis")
    print("-" * 50)

    # Systematic analysis
    analyze_systematic_deviations()

    # Specific examples
    demonstrate_best_response_function()

    # Visualization
    visualize_response_functions()

    return


def create_profit_landscape_visualization():
    """
    Create 3D visualization of profit landscape around Nash equilibrium
    """
    nash_eq = [-20, -28]

    # Create grid around Nash equilibrium
    p1_range = np.arange(-35, -5, 2)
    p2_range = np.arange(-40, -15, 2)
    P1, P2 = np.meshgrid(p1_range, p2_range)

    # Calculate profits for both players
    Profits_1 = np.zeros_like(P1)
    Profits_2 = np.zeros_like(P2)

    for i in range(len(p1_range)):
        for j in range(len(p2_range)):
            profit_1, _, _ = calculate_profit_change(P1[j, i], P2[j, i], player=1)
            profit_2, _, _ = calculate_profit_change(P1[j, i], P2[j, i], player=2)
            Profits_1[j, i] = profit_1
            Profits_2[j, i] = profit_2

    # Create visualization
    fig = plt.figure(figsize=(15, 5))

    # Player 1 profit landscape
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(P1, P2, Profits_1, cmap='RdYlBu', alpha=0.7)
    ax1.scatter([nash_eq[0]], [nash_eq[1]], [calculate_profit_change(nash_eq[0], nash_eq[1], 1)[0]],
                color='red', s=100, label='Nash Equilibrium')
    ax1.set_xlabel('Player 1 Price Change')
    ax1.set_ylabel('Player 2 Price Change')
    ax1.set_zlabel('Player 1 Profit Change (%)')
    ax1.set_title('Player 1 Profit Landscape')

    # Player 2 profit landscape
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(P1, P2, Profits_2, cmap='RdYlGn', alpha=0.7)
    ax2.scatter([nash_eq[0]], [nash_eq[1]], [calculate_profit_change(nash_eq[0], nash_eq[1], 2)[0]],
                color='red', s=100, label='Nash Equilibrium')
    ax2.set_xlabel('Player 1 Price Change')
    ax2.set_ylabel('Player 2 Price Change')
    ax2.set_zlabel('Player 2 Profit Change (%)')
    ax2.set_title('Player 2 Profit Landscape')

    # 2D contour plot showing both players
    ax3 = fig.add_subplot(133)

    # Plot Player 1's best response curve
    p1_best_responses = []
    for p2 in p2_range:
        best_profit = -float('inf')
        best_p1 = None
        for p1 in p1_range:
            profit, _, _ = calculate_profit_change(p1, p2, player=1)
            if profit > best_profit:
                best_profit = profit
                best_p1 = p1
        p1_best_responses.append(best_p1)

    # Plot Player 2's best response curve
    p2_best_responses = []
    for p1 in p1_range:
        best_profit = -float('inf')
        best_p2 = None
        for p2 in p2_range:
            profit, _, _ = calculate_profit_change(p1, p2, player=2)
            if profit > best_profit:
                best_profit = profit
                best_p2 = p2
        p2_best_responses.append(best_p2)

    ax3.plot(p2_range, p1_best_responses, 'b-', linewidth=2, label='Player 1 Best Response')
    ax3.plot(p2_best_responses, p1_range, 'r-', linewidth=2, label='Player 2 Best Response')
    ax3.scatter([nash_eq[1]], [nash_eq[0]], color='black', s=100, zorder=5,
                label='Nash Equilibrium', marker='*')

    ax3.set_xlabel('Player 2 Price Change')
    ax3.set_ylabel('Player 1 Price Change')
    ax3.set_title('Best Response Functions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demonstrate_key_insights():
    """
    Demonstrate key insights about Nash equilibrium and deviations
    """
    print("=== KEY INSIGHTS ABOUT NASH EQUILIBRIUM DEVIATIONS ===")
    print()

    print("1. UNILATERAL DEVIATION IS BAD")
    print("   If one player deviates alone, they typically get WORSE results")
    print("   Example: Player 1 at Nash (-20) vs deviating to -15 while Player 2 stays at -28")

    nash_profit_1, _, _ = calculate_profit_change(-20, -28, player=1)
    deviation_profit_1, _, _ = calculate_profit_change(-15, -28, player=1)

    print(f"   Nash profit: {nash_profit_1:.2f}%")
    print(f"   Deviation profit: {deviation_profit_1:.2f}%")
    print(f"   Loss from deviation: {deviation_profit_1 - nash_profit_1:.2f} percentage points")
    print()

    print("2. THE OTHER PLAYER BENEFITS FROM YOUR DEVIATION")
    print("   When you deviate (usually making things worse for yourself),")
    print("   your competitor often benefits!")

    nash_profit_2, _, _ = calculate_profit_change(-20, -28, player=2)
    competitor_benefit, _, _ = calculate_profit_change(-15, -28, player=2)

    print(f"   Player 2 at Nash: {nash_profit_2:.2f}%")
    print(f"   Player 2 when Player 1 deviates: {competitor_benefit:.2f}%")
    print(f"   Player 2's benefit: {competitor_benefit - nash_profit_2:.2f} percentage points")
    print()

    print("3. BOTH PLAYERS CAN IMPROVE WITH COORDINATED DEVIATION")
    print("   If both players could coordinate (form a cartel), they could both do better")
    print("   Example: Both players reduce price cuts by 50%")

    cartel_1 = -10  # 50% of -20
    cartel_2 = -14  # 50% of -28

    cartel_profit_1, _, _ = calculate_profit_change(cartel_1, cartel_2, player=1)
    cartel_profit_2, _, _ = calculate_profit_change(cartel_1, cartel_2, player=2)

    print(f"   Cartel Player 1: {cartel_profit_1:.2f}% (vs {nash_profit_1:.2f}% at Nash)")
    print(f"   Cartel Player 2: {cartel_profit_2:.2f}% (vs {nash_profit_2:.2f}% at Nash)")
    print(f"   Improvement: P1 = {cartel_profit_1 - nash_profit_1:.2f}, P2 = {cartel_profit_2 - nash_profit_2:.2f}")
    print()

    print("4. BUT CARTELS ARE UNSTABLE")
    print("   Each player has incentive to cheat on the cartel agreement")

    # Player 1 cheats on cartel by going back to Nash response
    cheat_profit_1, _, _ = calculate_profit_change(-20, cartel_2, player=1)

    print(f"   If Player 1 cheats on cartel: {cheat_profit_1:.2f}%")
    print(f"   Benefit from cheating: {cheat_profit_1 - cartel_profit_1:.2f} percentage points")
    print("   This is why cartels often collapse!")
    print()

    print("5. NASH EQUILIBRIUM IS 'STABLE'")
    print("   No player can improve by unilaterally changing strategy")
    print("   This makes it the predicted outcome of rational competition")


# === RUN THE COMPLETE ANALYSIS ===

if __name__ == "__main__":
    # Analyze deviation scenarios
    df1, df2, nash_eq = analyze_deviation_scenarios()

    # Analyze optimal responses
    response_results = analyze_optimal_responses()

    # Create visualizations
    create_profit_landscape_visualization()

    # Demonstrate key insights
    demonstrate_key_insights()

"""
Summary: The Answer to Your Question
YES, when one player deviates from Nash equilibrium, the other player can often take advantage - but there's a crucial catch:
The Deviating Player Usually Gets Hurt
When Player 1 deviates from -$20 to say -$15 (smaller price cut), they typically get worse profits than at Nash equilibrium.
The Non-Deviating Player Benefits
Player 2, staying at -$28, actually benefits from Player 1's deviation because Player 1 becomes less competitive.
Optimal Response Makes It Even Better
If Player 2 can respond optimally to Player 1's deviation, Player 2 can do even better than just staying at Nash.
The key insight is that unilateral deviations from Nash equilibrium typically hurts.
This is exactly why Nash equilibrium is "stable" - no player wants to deviate because they know they'll be worse off,
"""
