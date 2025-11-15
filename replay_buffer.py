# %%
# replay_buffer.py
import random
from collections import deque, namedtuple
import numpy as np
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity=100_000, seed=42):
        self.buf = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, s, a, r, ns, done):
        self.buf.append(Transition(
            np.asarray(s, dtype=np.float32),
            int(a),
            float(r),
            np.asarray(ns, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size=64):
        batch = random.sample(self.buf, batch_size)
        s  = np.stack([t.state for t in batch])
        a  = np.asarray([t.action for t in batch])
        r  = np.asarray([t.reward for t in batch], dtype=np.float32)
        ns = np.stack([t.next_state for t in batch])
        d  = np.asarray([t.done for t in batch], dtype=np.float32)
        return s, a, r, ns, d

    def __len__(self): return len(self.buf)


