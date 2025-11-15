🧠 eCommerce DQN Reinforcement Learning

본 프로젝트는 전자상거래(e-Commerce) 구매 이력 데이터를 기반으로,
할인 정책(Action)에 따른 고객 구매 반응을 학습하는 DQN (Deep Q-Network) 강화학습 모델 구현을 목표로 합니다.
Python + PyTorch 기반으로 환경 설계부터 학습, 평가, 시각화까지 전 과정을 포함합니다.

🚀 Project Overview
항목	내용
Algorithm	Deep Q-Network (DQN)
Goal	eCommerce 할인 정책 최적화
State (S)	recency_norm, frequency_norm, avg_price_norm, diversity_norm
Action (A)	{0: 무행동, 1: 5% 할인, 2: 10% 할인, 3: 20% 할인}
Reward (R)	구매 성공 시 +, 미구매 시 – 보상 (할인율 페널티 포함)
Transition (T, γ)	사용자별 구매 시퀀스 기반 전이, γ=0.95
📁 Directory Structure
📦 eCommerce_DQN
├── env.py              # 강화학습 환경 (MDP)
├── replay_buffer.py    # 경험 재현 버퍼 (Experience Replay)
├── dqn_agent.py        # Q-Network 및 DQN 에이전트
├── train_dqn.py        # 학습 루프 및 로깅
├── train_log.csv       # 학습 로그 (자동 생성)
├── log_visualizer.py   # 학습 시각화 (Return / Epsilon / Action)
├── states_30d.csv      # 전처리된 고객 상태 데이터
└── README.md           # 프로젝트 설명

⚙️ Setup & Run
1️⃣ Install dependencies
pip install torch pandas numpy matplotlib

2️⃣ Run preprocessing

states_30d.csv는 전처리 스크립트(build_states_final.py)로 생성됩니다.
(또는 제공된 샘플 데이터를 바로 사용 가능)

3️⃣ Train the DQN Agent
python train_dqn.py


학습이 진행되면 다음 파일이 자동 생성됩니다:

train_log.csv — 에피소드별 학습 로그

checkpoints/dqn_best.pt — 최고 성능 Q-network 파라미터

4️⃣ Visualize training results
python log_visualizer.py


Return, Epsilon Decay, Action Count 그래프가 출력됩니다.

📊 Results
항목	설명
Episode Return	학습이 진행될수록 안정적으로 수렴하며 보상이 증가함
Epsilon Decay	탐험(Exploration) → 활용(Exploitation) 전환 정상 작동
Action Distribution	무행동 대비 10~20% 쿠폰 사용 비율 점진적 증
	
🧩 Key Highlights

✅ Custom MDP environment for realistic eCommerce simulation
✅ Dynamic reward scaling with promo-cost penalty
✅ Experience Replay Buffer + Soft Target Update (τ=0.005)
✅ Auto logging & visualization pipeline integrated

🧠 Future Improvements

Double / Dueling DQN 구조 추가로 Q-value 안정화

Reward Engineering: 고객 재방문률, 이탈률 반영

Feature Expansion: 상품 카테고리, 시간대, 이벤트 정보 추가

Policy-based 알고리즘 실험 (PPO, A2C 등)

👨‍💻 Author

이재영 (Jae-Young Lee)
Graduate Student, AI · Data Science
📧 wodud4916@naver.com

📍 Sogang University
