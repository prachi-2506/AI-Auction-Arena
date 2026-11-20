import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 🎨 PAGE CONFIG + STYLE
# =============================

st.set_page_config(page_title="AI Auction Arena", layout="wide")

st.markdown("""
<style>
body { background-color: #0e1117; }
h1, h2, h3 { color: #00ffd5; }
.stButton>button {
    background-color: #00ffd5;
    color: black;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =============================
# 🧠 MODEL
# =============================

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.model(x)

device = torch.device("cpu")

model = Policy().to(device)
model.load_state_dict(torch.load("saved_model.pt", map_location=device))
model.eval()

def ai_bid(value):
    v = torch.tensor([[value]], dtype=torch.float32)
    with torch.no_grad():
        mu = model(v).item()
    noise = np.random.normal(0, 0.03)
    return np.clip(mu + noise, 0, 1)

# =============================
# 📊 SESSION STATE
# =============================

if "rounds" not in st.session_state:
    st.session_state.rounds = 0
    st.session_state.user_wins = 0
    st.session_state.ai_wins = 0
    st.session_state.user_profit = 0

# =============================
# 📌 SIDEBAR
# =============================

page = st.sidebar.radio(
    "🚀 Navigation",
    [
        "🎮 Play Game",
        "📘 Learn Theory",
        "🏆 Tournament",
        "📊 Benchmark",
        "📈 Strategy"
    ]
)

# =============================
# 🎮 PLAY GAME
# =============================

if page == "🎮 Play Game":

    st.title("🎮 AI Auction Arena")

    # 🧾 Instructions
    st.info("""
    ### 🎯 How to Play
    - You will be given a **private value** (how much item is worth to you)
    - You submit a **bid**
    - AI also submits a bid (unknown to you)
    - Highest bid wins
    - Winner pays their own bid

    👉 Your goal: **maximize profit = value − bid**
    """)

    st.warning("""
    ⚠️ Tips:
    - Bidding too high → you win but lose profit  
    - Bidding too low → you lose  
    - Optimal strategy ≈ **0.5 × your value**
    """)

    st.subheader("⚔️ Play Round")

    col1, col2 = st.columns(2)

    with col1:
        user_value = st.slider("💎 Your Value",0.0,1.0,0.5)

    with col2:
        user_bid = st.slider("💰 Your Bid",0.0,1.0,0.25)

    if st.button("🚀 Run Auction"):

        ai_value = np.random.rand()
        ai_bid_value = ai_bid(ai_value)

        st.session_state.rounds += 1

        st.markdown("---")
        st.subheader("📊 Result")

        st.write(f"Your Value: {user_value:.3f}")
        st.write(f"Your Bid: {user_bid:.3f}")
        st.write(f"AI Value: {ai_value:.3f}")
        st.write(f"AI Bid: {ai_bid_value:.3f}")

        if user_bid > ai_bid_value:
            profit = user_value - user_bid
            st.success(f"🎉 You Win! Profit = {profit:.3f}")
            st.session_state.user_wins += 1
            st.session_state.user_profit += profit

        elif ai_bid_value > user_bid:
            st.error("🤖 AI Wins!")
            st.session_state.ai_wins += 1

        else:
            st.warning("⚖️ Tie")

        st.markdown("---")
        st.info(f"🧠 Nash Suggestion: {0.5 * user_value:.3f}")

    st.subheader("🏆 Leaderboard")

    st.metric("Rounds", st.session_state.rounds)
    st.metric("Your Wins", st.session_state.user_wins)
    st.metric("AI Wins", st.session_state.ai_wins)

    if st.session_state.rounds > 0:
        win_rate = st.session_state.user_wins / st.session_state.rounds
        st.metric("Win Rate", f"{win_rate:.2%}")
        st.metric("Total Profit", f"{st.session_state.user_profit:.3f}")

# =============================
# 📘 THEORY PAGE
# =============================

elif page == "📘 Learn Theory":

    st.title("📘 Auction Theory")

    st.markdown("""
    ### 🎯 What is this game?

    This is a **First-Price Sealed-Bid Auction**:
    - Each player has a **private value**
    - Players submit bids **without seeing others**
    - Highest bidder wins
    - Winner pays their own bid

    ---
    ### 🧠 Optimal Strategy (Nash Equilibrium)

    👉 **Bid ≈ 0.5 × Value**

    Why?
    - Higher bid → higher chance to win
    - Lower bid → higher profit if you win
    - Nash balances both

    ---
    ### 🤖 AI Learning

    - AI was trained using **Reinforcement Learning**
    - It learned by playing against itself
    - Over time it discovered optimal strategy

    ---
    ### 💡 Concepts Used

    - Bayesian Game (private information)
    - Multi-Agent RL
    - Policy Gradient (REINFORCE)
    """)

# =============================
# 🏆 TOURNAMENT
# =============================

elif page == "🏆 Tournament":

    st.title("🏆 Tournament Mode")

    st.info("""
    ### 📌 How it works
    - AI plays multiple rounds automatically
    - User follows Nash strategy (0.5v)
    - Used to test long-term performance
    """)

    rounds = st.slider("Rounds",10,1000,100)

    if st.button("Run Tournament"):

        wins = 0
        profit = 0

        for _ in range(rounds):

            user_value = np.random.rand()
            ai_value = np.random.rand()

            user_bid = 0.5 * user_value
            ai_b = ai_bid(ai_value)

            if user_bid > ai_b:
                wins += 1
                profit += user_value - user_bid

        st.success(f"Win Rate: {wins/rounds:.2f}")
        st.write(f"Avg Profit: {profit/rounds:.3f}")

# =============================
# 📊 BENCHMARK
# =============================

elif page == "📊 Benchmark":

    st.title("📊 AI Benchmark")

    st.info("""
    Compare AI against different types of bidders:
    - Random → no strategy
    - Truthful → bids full value
    - Nash → optimal strategy
    """)

    rounds = st.slider("Rounds",100,10000,1000)

    def random_bid(v): return np.random.rand()
    def truthful_bid(v): return v
    def nash_bid(v): return 0.5 * v

    opponents = {
        "Random": random_bid,
        "Truthful": truthful_bid,
        "Nash": nash_bid
    }

    for name, strat in opponents.items():

        wins = 0
        profit = 0

        for _ in range(rounds):
            ai_v = np.random.rand()
            opp_v = np.random.rand()

            ai_b = ai_bid(ai_v)
            opp_b = strat(opp_v)

            if ai_b > opp_b:
                wins += 1
                profit += ai_v - ai_b

        st.write(f"### {name}")
        st.write(f"Win Rate: {wins/rounds:.2f}")
        st.write(f"Avg Profit: {profit/rounds:.3f}")
        st.markdown("---")

# =============================
# 📈 STRATEGY
# =============================

elif page == "📈 Strategy":

    st.title("📈 AI Strategy Visualization")

    st.info("""
    This graph shows how AI bids based on its value.
    Dashed line = theoretical Nash equilibrium.
    """)

    values = np.linspace(0,1,100)
    bids = [ai_bid(v) for v in values]
    nash = 0.5 * values

    fig, ax = plt.subplots()
    ax.plot(values, bids, label="AI Strategy")
    ax.plot(values, nash, '--', label="Nash")
    ax.set_xlabel("Value")
    ax.set_ylabel("Bid")
    ax.legend()

    st.pyplot(fig)