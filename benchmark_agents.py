import torch
import torch.nn as nn
import numpy as np


device = torch.device("cpu")


# Policy Network (same as training)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Load trained model

model = Policy().to(device)
model.load_state_dict(torch.load("saved_model.pt", map_location=device))
model.eval()

print("AI model loaded!")


# AI bidding function

def ai_bid(value):

    v = torch.tensor([[value]], dtype=torch.float32)

    with torch.no_grad():
        bid = model(v).item()

    return bid


# Opponent strategies

def random_bid(value):
    return np.random.rand()


def truthful_bid(value):
    return value


def nash_bid(value):
    return 0.5 * value


# Auction simulator

def run_simulation(opponent_bid_function, rounds=10000):

    ai_profit_total = 0
    ai_wins = 0

    for _ in range(rounds):

        ai_value = np.random.rand()
        opp_value = np.random.rand()

        ai_b = ai_bid(ai_value)
        opp_b = opponent_bid_function(opp_value)

        if ai_b > opp_b:

            ai_wins += 1
            profit = ai_value - ai_b
            ai_profit_total += profit

        elif opp_b > ai_b:

            pass

        else:
            pass

    avg_profit = ai_profit_total / rounds
    win_rate = ai_wins / rounds

    return avg_profit, win_rate


# Run experiments

opponents = {
    "Random Bidder": random_bid,
    "Truthful Bidder": truthful_bid,
    "Nash Bidder": nash_bid
}

print("\n===== BENCHMARK RESULTS =====\n")

for name, strategy in opponents.items():

    profit, winrate = run_simulation(strategy)

    print(f"{name}")
    print(f"Average Profit: {profit:.4f}")
    print(f"Win Rate: {winrate:.2f}")
    print("-------------------------")