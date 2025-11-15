# %%
# build_states_final.py
# pip install pandas numpy
import pandas as pd
import numpy as np

path = r"C:\Users\admin\Desktop\강화학습_과제\Data\states_30d.csv"

df = pd.read_csv(path)

df.head()

# %%
# ===============================================================
# ecommerce_env.py
# ---------------------------------------------------------------
# DQN 학습용 eCommerce 시뮬레이터 환경
# ---------------------------------------------------------------
# (S, A, R, T, γ)
# S: [recency, frequency, avg_price, diversity]
# A: {0=무행동, 1=5%쿠폰, 2=10%쿠폰, 3=20%쿠폰}
# R: 구매 성공/실패 기반
# T: 다음 시점 고객 데이터로 전이
# ===============================================================

import numpy as np
import pandas as pd
import random

class ECommerceEnv:
    def __init__(self, path, gamma=0.9):
        """
        eCommerce 강화학습 환경 초기화
        :param csv_path: 전처리된 states_30d.csv 파일 경로
        :param gamma: 할인율
        """
        self.gamma = gamma
        self.data = pd.read_csv(path)
        self.users = self.data['user_id'].unique()
        
        # 전체 기준값 (보상 정규화용)
        self.median_price = max(1.0, float(self.data["price"].median()))
        
        # state feature 정의
        self.state_cols = [
            "recency_norm", "frequency_norm", "avg_price_norm", "diversity_norm"
        ]
        
        # 환경 내부 상태 변수
        self.current_user = None
        self.current_index = None
        self.done = False

    # -----------------------------------------------------------
    def reset(self):
        """
        새로운 고객 에피소드 시작
        :return: 초기 상태 (np.array)
        """
        # 랜덤 고객 선택
        self.current_user = random.choice(self.users)
        user_df = self.data[self.data['user_id'] == self.current_user].sort_values('event_time')
        
        self.user_df = user_df.reset_index(drop=True)
        self.current_index = 0
        self.done = False
        
        # 첫 번째 상태 반환
        return self._get_state(self.current_index)

    # -----------------------------------------------------------
    def step(self, action):
        """
        현재 상태에서 action을 수행하고 reward, 다음 상태 반환
        :param action: 정수형 (0~3)
        :return: (next_state, reward, done, info)
        """
        if self.done:
            raise Exception("Episode has ended. Please call reset().")

        # 현재 행
        row = self.user_df.iloc[self.current_index]
        price = float(row["price"])
        
        # Reward 계산
        reward = self._calculate_reward(action, price)
        
        # 다음 상태로 이동
        self.current_index += 1
        if self.current_index >= len(self.user_df):
            self.done = True
            next_state = np.zeros(len(self.state_cols))  # terminal state
        else:
            next_state = self._get_state(self.current_index)
        
        info = {
            "user_id": self.current_user,
            "step_index": self.current_index,
            "action": action,
            "reward": reward,
        }
        
        return next_state, reward, self.done, info

    # -----------------------------------------------------------
    def _get_state(self, idx):
        """현재 인덱스의 상태 벡터 반환"""
        row = self.user_df.iloc[idx]
        return row[self.state_cols].to_numpy(dtype=np.float32)
    
    # -----------------------------------------------------------
    def _calculate_reward(self, action, price):
        """
        Reward 계산 로직
        - 구매 성공(가정): 확률적으로 보상 반환
        - 실제 구매 데이터 기반이므로 price>0 시 구매 발생으로 간주
        """
        if price <= 0:
            # 미구매
            return -0.1
        else:
            # 구매 발생
            base_reward = price / self.median_price
            promo_cost = 0.0

            # 할인율에 따라 비용 페널티 부여
            if action == 1:
                promo_cost = 0.05
            elif action == 2:
                promo_cost = 0.10
            elif action == 3:
                promo_cost = 0.20

            # 최종 보상 (구매 보상 - 할인비용)
            reward = base_reward * (1 - promo_cost)
            return reward
    
    # -----------------------------------------------------------
    def render(self):
        """현재 상태 간단히 출력"""
        print(f"[User {self.current_user}] Step={self.current_index}, Done={self.done}")

# ===============================================================
# 실행 테스트용 코드
# ===============================================================
if __name__ == "__main__":
    env = ECommerceEnv(r"C:\Users\admin\Desktop\강화학습_과제\Data\states_30d.csv")
    
    # 에피소드 시작
    state = env.reset()
    print("초기 상태:", state)
    
    total_reward = 0
    done = False
    
    while not done:
        action = np.random.choice([0, 1, 2, 3])  # 랜덤 액션
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"A:{action}, R:{reward:.3f}, Done:{done}")
        
    print(f"총 보상합계: {total_reward:.2f}")


# %%



