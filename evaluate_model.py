import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu")

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

model = Policy().to(device)
model.load_state_dict(torch.load("saved_model.pt", map_location=device))
model.eval()

values = torch.linspace(0,1,100).unsqueeze(1)

with torch.no_grad():
    bids = model(values).squeeze().numpy()

values = values.squeeze().numpy()
nash = 0.5 * values

plt.plot(values, bids, label="Learned Strategy")
plt.plot(values, nash, '--', label="Nash Equilibrium")
plt.xlabel("Value")
plt.ylabel("Bid")
plt.title("Learned Auction Strategy")
plt.legend()
plt.show()