# AI Auction Arena

A first-price sealed-bid auction simulator in which two neural-network agents learn bidding
strategies from scratch through self-play, and are benchmarked against the theoretical
game-theoretic optimum.

**Live demo:** https://ai-auction-arena.streamlit.app/

---

## The problem

In a first-price sealed-bid auction, each bidder privately observes their own valuation of an
item, submits a single bid without seeing anyone else's, and the highest bidder wins and pays
exactly what they bid. Profit is `value − bid` for the winner and zero for everyone else.

This creates a genuine strategic tension: bid your true value and you win often but profit
nothing; shade your bid too far and you profit when you win but rarely win. The question is
whether agents with no prior knowledge of auction theory can discover the right amount of
shading purely from experience.

For the symmetric two-bidder case with values drawn independently from a uniform distribution,
auction theory gives the Bayes-Nash equilibrium strategy in closed form:

```
b(v) = 0.5 · v
```

That closed form is what makes this a useful testbed — there is a known right answer to measure
learned behaviour against.

---

## What this project does

Two independent policy networks play repeated auctions against each other. Each receives a
private value, samples a bid, and is rewarded by the auction outcome. Over 120,000 training
episodes the agents adjust their policies by policy gradient, and the learned bidding function
is compared against `b(v) = 0.5v` along with random and truthful baselines.

A second, independent method — discretised regret matching — is implemented separately so the
two approaches to strategy learning can be compared on the same game.

---

## Approach and design decisions

**Policy network.** A small MLP: `1 → 64 → 64 → 1`, ReLU activations, sigmoid output. The
sigmoid matters — values are drawn from `U(0,1)`, so bounding the output to `[0,1]` makes
out-of-range bids structurally impossible rather than something the agent must learn to avoid.

**Stochastic policy for exploration.** The network emits the *mean* of a Gaussian
(`σ = 0.05`); the actual bid is sampled from it and clamped to `[0,1]`. A deterministic policy
would have no exploration signal in a continuous action space, so the gradient needs the
sampled action and its log-probability.

**REINFORCE with advantage normalisation.** The policy-gradient loss is
`−(log π(b|v) · A)` where the advantage is the batch reward standardised to zero mean and unit
variance. Raw auction rewards are sparse and heavily skewed — a losing bid pays exactly zero —
so without normalisation the gradient magnitude swings wildly between batches.

**Entropy bonus (`0.01`).** Counteracts premature collapse onto a narrow bid range before the
agent has explored enough of the strategy space.

**Alternating self-play.** The central difficulty in multi-agent learning is that your opponent
is also learning, so the environment is non-stationary — each agent is chasing a moving target.
This implementation freezes one agent for 2,000-episode phases while the other updates, then
alternates. The frozen agent acts as a temporarily stationary opponent, which stabilises
training considerably compared with updating both simultaneously.

**Hyperparameters.** 120,000 episodes, batch size 512, Adam at `5e-4`, phase length 2,000.

---

## Results and how to read them

After training, the learned bidding function is evaluated on a grid of 100 values and compared
against the analytical benchmark by mean squared error:

```
MSE vs Nash (Agent 1): printed at end of training
MSE vs Nash (Agent 2): printed at end of training
```

**What this does and does not show.** A low MSE means the learned bidding function lies close
to the theoretical equilibrium strategy over the sampled value range. It is an empirical
distance measurement, not a proof. This project does not establish convergence to a Nash
equilibrium in any formal sense — no equilibrium certificate, no exploitability bound, and no
guarantee the policy is stable under further training. Treat the number as evidence that
self-play recovered approximately the right shading behaviour, nothing stronger.

`benchmark_agents.py` gives the complementary view: how the trained agent actually performs in
head-to-head play against random, truthful and Nash-strategy opponents.

---

## Files

| File | Purpose |
|---|---|
| `train_self_rl.py` | Self-play training loop; saves `saved_model.pt` |
| `evaluate_model.py` | Plots the learned bidding function against the Nash benchmark |
| `benchmark_agents.py` | Head-to-head evaluation vs random, truthful and Nash bidders |
| `regret_minimization.py` | Independent regret-matching experiment for comparison |
| `auction_game.py` | Terminal game: play auctions against the trained agent |
| `streamlit_app.py` | Web UI — play, theory, tournament, benchmark and strategy views |
| `saved_model.pt` | Trained PyTorch state dict (Agent 1) |

---

## Running it

Python 3.10+.

```bash
pip install -r requirements.txt
```

```bash
streamlit run streamlit_app.py      # web interface
python train_self_rl.py             # retrain (overwrites saved_model.pt)
python evaluate_model.py            # strategy plot vs benchmark
python benchmark_agents.py          # head-to-head results
python regret_minimization.py       # regret-matching comparison
python auction_game.py              # terminal game
```

Scripts that load the checkpoint expect `saved_model.pt` in the working directory.

---

## Limitations

- Two bidders only, symmetric, with values from `U(0,1)`. Asymmetric valuations, reserve
  prices, more than two bidders, and other auction formats (second-price, all-pay) are not
  implemented — each would change the equilibrium and require a different benchmark.
- `train_self_rl.py` saves only Agent 1's weights; Agent 2 is discarded after training.
- This is a research and learning simulator, not a production auction or trading system.

---

## Possible extensions

Second-price and all-pay formats (where the equilibrium strategies differ sharply), asymmetric
value distributions, N > 2 bidders, and measuring exploitability against a best-response
opponent rather than distance from a closed-form benchmark.
