# %%
import pandas as pd
import numpy as np


df = pd.read_csv(r'C:\Users\admin\Desktop\강화학습_과제\Data\train_log.csv')

df.head()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import os

# 로그 파일 경로
log_path = r"C:\Users\admin\Desktop\강화학습_과제\Data\train_log.csv"
if not os.path.exists(log_path):
    raise FileNotFoundError(f"로그 파일이 없습니다: {log_path}")

# 데이터 불러오기
df = pd.read_csv(log_path)
print(df.head())

# ───────────────────────────────────────────
# 1️⃣ Return 곡선
# ───────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["return"], label="Episode Return", alpha=0.8)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("DQN Training Progress")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ───────────────────────────────────────────
# 2️⃣ Epsilon 감소 추이
# ───────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(df["episode"], df["epsilon"], color="orange", label="Epsilon")
plt.xlabel("Episode")
plt.ylabel("Epsilon (Exploration Rate)")
plt.title("Epsilon Decay")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ───────────────────────────────────────────
# 3️⃣ 행동 선택 비율
# ───────────────────────────────────────────
acts = [c for c in df.columns if c.startswith("act")]
act_sums = df[acts].sum()
plt.figure(figsize=(8, 4))
plt.bar(acts, act_sums, color="skyblue")
plt.title("Total Action Counts Across Training")
plt.xlabel("Action")
plt.ylabel("Count")
plt.tight_layout()
plt.show()



