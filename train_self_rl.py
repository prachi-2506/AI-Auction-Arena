import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt



# Hyperparameters


episodes = 120000
batch_size = 512
learning_rate = 0.0005

sigma = 0.05           
entropy_coeff = 0.01   

phase_length = 2000    

device = torch.device("cpu")



# Policy Network


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



# Initialize Agents


agent1 = Policy().to(device)
agent2 = Policy().to(device)

opt1 = optim.Adam(agent1.parameters(), lr=learning_rate)
opt2 = optim.Adam(agent2.parameters(), lr=learning_rate)



# Learning Curve Storage


reward_history_1 = []
reward_history_2 = []

plot_interval = 500



# Training Loop (Self-Play RL)


for episode in range(episodes):

    # sample private values
    v1 = torch.rand(batch_size, 1).to(device)
    v2 = torch.rand(batch_size, 1).to(device)

    # mean bids from policy
    mu1 = agent1(v1)
    mu2 = agent2(v2)

    dist1 = torch.distributions.Normal(mu1, sigma)
    dist2 = torch.distributions.Normal(mu2, sigma)

    # sample bids
    b1 = torch.clamp(dist1.sample(), 0, 1)
    b2 = torch.clamp(dist2.sample(), 0, 1)

    log_prob1 = dist1.log_prob(b1)
    log_prob2 = dist2.log_prob(b2)

    # auction outcome
    win1 = (b1 > b2).float()
    win2 = (b2 > b1).float()

    # rewards
    r1 = win1 * (v1 - b1)
    r2 = win2 * (v2 - b2)

    # store learning curve
    if episode % plot_interval == 0:
        reward_history_1.append(r1.mean().item())
        reward_history_2.append(r2.mean().item())

    # advantage normalization
    adv1 = (r1 - r1.mean()) / (r1.std() + 1e-8)
    adv2 = (r2 - r2.mean()) / (r2.std() + 1e-8)

    # entropy bonus
    entropy1 = dist1.entropy().mean()
    entropy2 = dist2.entropy().mean()

    # policy gradient loss
    loss1 = -(log_prob1 * adv1.detach()).mean() - entropy_coeff * entropy1
    loss2 = -(log_prob2 * adv2.detach()).mean() - entropy_coeff * entropy2


    
    # Alternating Self-Play Training
    

    phase = (episode // phase_length) % 2

    if phase == 0:
       
        opt1.zero_grad()
        loss1.backward()
        opt1.step()

    else:
       
        opt2.zero_grad()
        loss2.backward()
        opt2.step()


    if episode % 5000 == 0:
        print(f"Episode {episode}")


print("\nTraining complete!")



# Plot Learning Curve


episodes_axis = np.arange(len(reward_history_1)) * plot_interval

plt.figure(figsize=(8,6))
plt.plot(episodes_axis, reward_history_1, label="Agent 1 Reward")
plt.plot(episodes_axis, reward_history_2, label="Agent 2 Reward")

plt.xlabel("Training Episode")
plt.ylabel("Average Reward")
plt.title("Learning Curve: Self-Play RL Training")
plt.legend()
plt.show()



# Evaluation & Convergence


value_grid = torch.linspace(0, 1, 100).unsqueeze(1)

with torch.no_grad():
    learned1 = agent1(value_grid).squeeze().numpy()
    learned2 = agent2(value_grid).squeeze().numpy()

nash = 0.5 * value_grid.squeeze().numpy()

# convergence metric
mse1 = np.mean((learned1 - nash) ** 2)
mse2 = np.mean((learned2 - nash) ** 2)

print(f"\nMSE vs Nash (Agent 1): {mse1:.6f}")
print(f"MSE vs Nash (Agent 2): {mse2:.6f}")



# Strategy Plot


plt.figure(figsize=(8,6))

plt.plot(value_grid.numpy(), learned1, label="Agent 1")
plt.plot(value_grid.numpy(), learned2, label="Agent 2")
plt.plot(value_grid.numpy(), nash, "--", label="Nash (0.5v)")

plt.xlabel("Value")
plt.ylabel("Bid")
plt.title("Learned Strategy vs Nash Benchmark")

plt.legend()
plt.show()



# Save Model


torch.save(agent1.state_dict(), "saved_model.pt")

print("\nModel saved as saved_model.pt")