import numpy as np
import torch
import torch.nn as nn
import os

device = torch.device("cpu")

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



model = Policy().to(device)

if os.path.exists("saved_model.pt"):
    model.load_state_dict(torch.load("saved_model.pt", map_location=device))
    model.eval()
    print("Trained AI model loaded successfully!")
    use_ml_model = True
else:
    print("No trained model found. Using Nash equilibrium strategy.")
    use_ml_model = False



def ai_bid(ai_value):

    if use_ml_model:

        with torch.no_grad():
            v = torch.tensor([[ai_value]], dtype=torch.float32).to(device)
            mu = model(v).item()
        noise = np.random.normal(0, 0.03)

        bid = mu + noise

        return np.clip(bid, 0, 1)

    else:
        return 0.5 * ai_value





class GameStats:

    def __init__(self):
        self.rounds = 0
        self.user_wins = 0
        self.ai_wins = 0
        self.user_profit_total = 0

    def update(self, user_win, profit):

        self.rounds += 1

        if user_win:
            self.user_wins += 1
            self.user_profit_total += profit
        else:
            self.ai_wins += 1

    def display(self):

        print("\n===== LEADERBOARD =====")
        print(f"Total Rounds: {self.rounds}")
        print(f"Your Wins: {self.user_wins}")
        print(f"AI Wins: {self.ai_wins}")

        if self.rounds > 0:
            win_rate = (self.user_wins / self.rounds) * 100
            avg_profit = self.user_profit_total / self.rounds
        else:
            win_rate = 0
            avg_profit = 0

        print(f"Your Win Rate: {round(win_rate,2)}%")
        print(f"Average Profit per Round: {round(avg_profit,3)}")
        print("=======================\n")





def run_auction(stats):

    print("\n===== FIRST-PRICE SEALED-BID AUCTION =====")

    user_value = float(input("Enter your private value (0 to 1): "))

    while user_value < 0 or user_value > 1:
        user_value = float(input("Value must be between 0 and 1. Try again: "))
    user_bid = float(input("Enter your bid (0 to 1): "))

    while user_bid < 0 or user_bid > 1:
        user_bid = float(input("Bid must be between 0 and 1. Try again: "))

  
    ai_value = np.random.rand()
    ai_bid_value = ai_bid(ai_value)

    print("\n--- Auction Result ---")

    print(f"Your Value: {round(user_value,3)}")
    print(f"Your Bid: {round(user_bid,3)}")

    print(f"AI Value: {round(ai_value,3)}")
    print(f"AI Bid: {round(ai_bid_value,3)}")

    if user_bid > ai_bid_value:

        print("\nYou WIN!")

        profit = user_value - user_bid

        print(f"Your Profit: {round(profit,3)}")

        stats.update(True, profit)

    elif ai_bid_value > user_bid:

        print("\nAI Wins.")

        print("Your Profit: 0")

        stats.update(False, 0)

    else:

        print("\nIt's a tie! No winner.")

        stats.update(False, 0)


    suggested_bid = 0.5 * user_value

    print("\n--- Nash Equilibrium Suggestion ---")

    print(f"Optimal Nash Bid: {round(suggested_bid,3)}")



# Game Loop


if __name__ == "__main__":

    stats = GameStats()

    print("\n🎮 Welcome to the AI Auction Arena 🎮")

    while True:

        run_auction(stats)

        show_stats = input("\nView leaderboard? (y/n): ")

        if show_stats.lower() == 'y':
            stats.display()

        again = input("Play another round? (y/n): ")

        if again.lower() != 'y':
            break

    print("\nFinal Leaderboard:")

    stats.display()

    print("Thanks for playing the AI Auction Game!")