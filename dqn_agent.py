# %%
# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x): return self.net(x)

class DQNAgent:
    def __init__(
        self,
        state_dim=4, action_dim=4, hidden=128,
        lr=1e-3, gamma=0.90, tau=0.005,  # tau for soft target update
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNet(state_dim, action_dim, hidden).to(self.device)
        self.t = QNet(state_dim, action_dim, hidden).to(self.device)
        self.t.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.loss_fn = nn.SmoothL1Loss()  # Huber

    @torch.no_grad()
    def act(self, state, epsilon: float):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(s)  # [1, A]
        return int(q.argmax(dim=1).item())

    def update(self, batch, grad_clip=5.0):
        s, a, r, ns, d = batch
        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d  = torch.tensor(d,  dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.q(s).gather(1, a)  # [B,1]

        # target: r + gamma * max_a' Q_target(ns, a') * (1 - done)
        with torch.no_grad():
            q_next_max = self.t(ns).max(dim=1, keepdim=True).values
            target = r + (1.0 - d) * self.gamma * q_next_max

        loss = self.loss_fn(q_sa, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), grad_clip)
        self.opt.step()

        # soft target update
        with torch.no_grad():
            for tp, sp in zip(self.t.parameters(), self.q.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)
        return float(loss.item())

# %%

class DQNAgent:
    def __init__(
        self,
        state_dim=4, action_dim=4, hidden=128,
        lr=1e-3, gamma=0.90, tau=0.005,  # tau for soft target update
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNet(state_dim, action_dim, hidden).to(self.device)
        self.t = QNet(state_dim, action_dim, hidden).to(self.device)
        self.t.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.loss_fn = nn.SmoothL1Loss()  # Huber

    @torch.no_grad()
    def act(self, state, epsilon: float):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(s)  # [1, A]
        return int(q.argmax(dim=1).item())

    def update(self, batch, grad_clip=5.0):
        s, a, r, ns, d = batch
        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d  = torch.tensor(d,  dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.q(s).gather(1, a)  # [B,1]

        # target: r + gamma * max_a' Q_target(ns, a') * (1 - done)
        with torch.no_grad():
            q_next_max = self.t(ns).max(dim=1, keepdim=True).values
            target = r + (1.0 - d) * self.gamma * q_next_max

        loss = self.loss_fn(q_sa, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), grad_clip)
        self.opt.step()

        # soft target update
        with torch.no_grad():
            for tp, sp in zip(self.t.parameters(), self.q.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)
        return float(loss.item())


