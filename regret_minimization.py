import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Discretization
num_values = 40
num_bids = 40

values = np.linspace(0, 1, num_values)
bids = np.linspace(0, 1, num_bids)

iterations = 2000

# Regret and strategy tables
regret_sum = np.zeros((num_values, num_bids))
strategy_sum = np.zeros((num_values, num_bids))

def regret_matching(regrets):
    positive_regrets = np.maximum(regrets, 0)
    if positive_regrets.sum() > 0:
        return positive_regrets / positive_regrets.sum()
    else:
        return np.ones(len(regrets)) / len(regrets)

# Initialize uniform strategy
strategy = np.ones((num_values, num_bids)) / num_bids

for it in range(iterations):

    new_strategy = np.zeros_like(strategy)

    for v_idx, v in enumerate(values):

        # Expected utility of each possible bid
        util = np.zeros(num_bids)

        for b_idx, b in enumerate(bids):

            expected_payoff = 0

            for v2_idx, v2 in enumerate(values):
                for b2_idx, b2 in enumerate(bids):

                    prob = strategy[v2_idx, b2_idx] / num_values

                    if b > b2:
                        payoff = v - b
                    elif b < b2:
                        payoff = 0
                    else:
                        payoff = 0

                    expected_payoff += prob * payoff

            util[b_idx] = expected_payoff

        # Current strategy payoff
        strat_payoff = np.dot(strategy[v_idx], util)

        # Update regrets
        regret_sum[v_idx] += util - strat_payoff

        # Update strategy via regret matching
        new_strategy[v_idx] = regret_matching(regret_sum[v_idx])

        # Accumulate average strategy
        strategy_sum[v_idx] += new_strategy[v_idx]

    strategy = new_strategy

    if it % 100 == 0:
        print("Iteration", it)

print("Training complete!")

# Compute average strategy
avg_strategy = strategy_sum / iterations

# Extract expected bid for each value
expected_bids = np.zeros(num_values)

for v_idx in range(num_values):
    expected_bids[v_idx] = np.dot(avg_strategy[v_idx], bids)

# Plot
plt.figure(figsize=(8,5))
plt.plot(values, expected_bids, label="CFR Learned Strategy")
plt.plot(values, 0.5 * values, '--', label="Nash (0.5v)")
plt.xlabel("Value")
plt.ylabel("Bid")
plt.title("Regret Minimization Converging to Nash")
plt.legend()
plt.show()
