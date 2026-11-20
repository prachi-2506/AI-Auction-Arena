# AI Auction Arena

## Project overview
AI Auction Arena is a first-price sealed-bid auction simulator. It trains two independent neural-network bidding agents through self-play using a REINFORCE-style policy-gradient approach. Each agent receives a private item value, samples a bid, competes in an auction, and receives a reward equal to value minus bid if it wins (zero if it loses).

The project also includes:
- A command-line auction game against the trained AI
- A Streamlit interface for playing, learning auction theory, tournaments, benchmarks, and strategy visualization
- A separate regret-minimization experiment
- Evaluation and benchmarking scripts
- A saved PyTorch model checkpoint

The scripts compare the learned bidding strategy with the theoretical benchmark b(v) = 0.5v for the symmetric two-bidder, independent uniform-value first-price auction.

## Files
- `train_self_rl.py` — trains two policy networks using alternating self-play updates, policy-gradient losses, entropy regularization, and reward normalization. Saves `saved_model.pt`.
- `evaluate_model.py` — loads the trained model and plots its bidding function against the Nash benchmark.
- `auction_game.py` — terminal-based interactive auction game against the AI, with a leaderboard.
- `streamlit_app.py` — web UI with Play Game, Learn Theory, Tournament, Benchmark, and Strategy pages.
- `regret_minimization.py` — discretized regret-matching experiment and strategy comparison plot.
- `benchmark_agents.py` — evaluates the trained AI against random, truthful, and Nash-strategy opponents.
- `saved_model.pt` — trained PyTorch state-dictionary checkpoint used by the evaluation and game applications.

## Setup
Use Python 3.10+ (a compatible PyTorch installation is required).

Install dependencies:
```bash
pip install torch numpy matplotlib streamlit
```

Run training:
```bash
python train_self_rl.py
```

Run evaluation:
```bash
python evaluate_model.py
```

Run the terminal game:
```bash
python auction_game.py
```

Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

Run regret minimization:
```bash
python regret_minimization.py
```

Run benchmark:
```bash
python benchmark_agents.py
```

Keep `saved_model.pt` in the same working directory as scripts that load it. Running the training script will overwrite the checkpoint with the newly trained Agent 1 model.

## Important implementation note
This is a research/learning simulator, not a production auction or financial trading system. The training code alternates updates between agents and uses sampled, clipped Normal bids. The evaluation scripts provide empirical comparisons; they do not by themselves prove convergence to a Nash equilibrium.
