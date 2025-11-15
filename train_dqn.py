# %%
import sys
import os
import numpy as np
import csv

# ───────────────────────────────────────────
# 로깅 유틸 함수
# ───────────────────────────────────────────
def init_action_counter(n_actions=4):
    """에피소드별 액션 선택 횟수 기록용"""
    return {a: 0 for a in range(n_actions)}

def write_log_row(csv_path, row_dict, header=None):
    """dict 형태의 로그를 CSV 파일에 저장"""
    first_write = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header or list(row_dict.keys()))
        if first_write:
            w.writeheader()
        w.writerow(row_dict)

# ───────────────────────────────────────────
# 1️⃣ 현재 파일 위치를 import 경로에 추가
# ───────────────────────────────────────────
sys.path.append(r"C:\Users\admin\Desktop\강화학습_과제")

# ───────────────────────────────────────────
# 2️⃣ 필요한 모듈 불러오기
# ───────────────────────────────────────────
from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent
from env import ECommerceEnv

# ───────────────────────────────────────────
# 3️⃣ 데이터 경로 지정
# ───────────────────────────────────────────
CSV_PATH = r"C:\Users\admin\Desktop\강화학습_과제\Data\states_30d.csv"


# ───────────────────────────────────────────
# 4️⃣ 학습 함수 정의
# ───────────────────────────────────────────
def train(
    episodes=500,
    warmup_steps=2_000,
    batch_size=128,
    buffer_capacity=100_000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=30_000,
    eval_every=50
):
    env = ECommerceEnv(CSV_PATH, gamma=0.90)
    agent = DQNAgent(state_dim=4, action_dim=4, hidden=128, lr=1e-3, gamma=0.90, tau=0.005)
    rb = ReplayBuffer(capacity=buffer_capacity)

    eps = epsilon_start
    eps_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)

    total_steps = 0
    best_eval = -1e9

    log_path = "train_log.csv"   # 로그 저장 파일명
    n_actions = 4                # 액션 개수

    # ───────────────────────────────────────────
    # 학습 루프 시작
    # ───────────────────────────────────────────
    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0
        act_count = init_action_counter(n_actions)

        while not done:
            # ε-greedy 행동 선택
            a = agent.act(s, epsilon=eps)
            ns, r, done, info = env.step(a)
            rb.push(s, a, r, ns, done)
            s = ns
            ep_ret += r
            ep_len += 1
            act_count[a] += 1
            total_steps += 1

            # Warmup 지나면 학습
            if len(rb) >= max(batch_size, warmup_steps):
                batch = rb.sample(batch_size)
                loss = agent.update(batch)

            # epsilon 선형 감소
            if eps > epsilon_end:
                eps = max(epsilon_end, eps - eps_decay)

        # ✅ 한 에피소드 끝날 때 로그 기록
        row = {"episode": ep, "return": round(ep_ret, 4), "length": ep_len, "epsilon": round(eps, 4)}
        for a_id in range(n_actions):
            row[f"act{a_id}"] = act_count[a_id]
        write_log_row(log_path, row, header=list(row.keys()))

        # 평가 루틴
        if ep % eval_every == 0:
            avg = evaluate(env, agent, episodes=10)
            print(f"[Eval] ep={ep:4d}  avg_return={avg:.3f}  eps={eps:.3f}  buffer={len(rb)}")
            if avg > best_eval:
                best_eval = avg
                os.makedirs("checkpoints", exist_ok=True)
                import torch
                torch.save(agent.q.state_dict(), os.path.join("checkpoints", "dqn_best.pt"))
                print("  ↳ ✅ best model saved")

        print(f"[Train] ep={ep:4d}  return={ep_ret:.3f}  len={ep_len:3d}  eps={eps:.3f}  buffer={len(rb)}")

    print("✅ training done. best eval:", best_eval)


# ───────────────────────────────────────────
# 5️⃣ 평가 함수
# ───────────────────────────────────────────
def evaluate(env, agent, episodes=5):
    total = 0.0
    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(s, epsilon=0.0)  # greedy
            ns, r, done, _ = env.step(a)
            s = ns
            ep_ret += r
        total += ep_ret
    return total / episodes


# ───────────────────────────────────────────
# 6️⃣ 실행 시작
# ───────────────────────────────────────────
if __name__ == "__main__":
    train(
        episodes=200,          # 처음엔 200~400으로 가볍게
        warmup_steps=2_000,    # 리플레이 버퍼 워밍업
        batch_size=128,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=30_000,
        eval_every=50
    )


# %%



