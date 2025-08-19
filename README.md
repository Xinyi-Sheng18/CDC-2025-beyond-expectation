# CDC-2025-beyond-expectation
This repository contains the official code for our CDC 2025 paper:
> **Beyond expected value: geometric mean optimization for long-term policy performance in reinforcement learning**
> Xinyi Sheng and Dominik Baumann
> Accepted at **IEEE CDC 2025**

## 📁 Overview

This project includes three experimental components:

1. **CartPole (gym-based)**
2. **LunarLander (gym-based with DQN)**
3. **CoinToss (toy example)**  
   Adapted from [Baumann et al., TMLR 2025](https://github.com/baumanndominik/ergodic_rl)

The core idea is to investigate multi-step Q-learning under **modified returns**, and study the effect of return modification via a `λ`-parameterized method.

---

## 🚀 Experiments

###  CartPole (Discrete Environment)

- Environment discretized manually.
- `λ = 0` represents standard multi-step bootstrapping baseline (with overlapping sampling).


### LunarLander (DQN + Our Method)

- Built upon the original DQN architecture from  
[katnoria's implementation](https://www.katnoria.com/nb_dqn_lunar/).
- Our method modifies the multi-step return without overlapping sampling.
- A baseline (`e=0`, `λ=0`) is also provided and used as the red dashed line in result plots.

### CoinToss Toy Example

- Environment and PPO baseline adapted from:
  > Baumann, D., et al.  
  > *"Reinforcement learning with non-ergodic reward increments: robustness via ergodicity transformations"*, TMLR 2025 (arXiv)  
  > [GitHub Repo](https://github.com/baumanndominik/ergodic_rl)



---

## 📊 Results (Visualizations)

All evaluation results are under the `results/` directory.

- For each policy (trained under a different `λ`), 100 episodes are tested.
- The distribution of **cumulative rewards** is plotted as a **density graph**, each dot represents a cumulative reward from one episode
- Two key statistics:
- **Median** (black line; shown in paper)
- **Mean** (black line; supplementary only, not shown in paper)

---

## 📂 Folder Structure

```text
experiments/
├── cartpole_5steps/        # CartPole training/testing
├── lunarlander_2steps/     # LunarLander training/testing + baselines
├── cointoss/               # CoinToss environment + notebook

results/                    # Policy evaluation visualizations (png/tex)

requirements/               # requirements_gym.txt & requirements_cointoss.txt

README.md
LICENSE


---


## 📝 License & Acknowledgments

This code is licensed under the MIT License.

### 🔗 Code Attribution 

The CoinToss.py environment and PPO baseline code are adapted from:
https://github.com/baumanndominik/ergodic_rl
by Baumann et al., TMLR 2025 (MIT License)

The DQN implementation used in LunarLander is adapted from:  
https://www.katnoria.com/nb_dqn_lunar/

### 💰 Funding

This work is supported by the **Finnish Ministry of Education and Culture** through the  
**Intelligent Work Machines Doctoral Education Pilot Program (IWM)**  
Grant No. **VN/3137/2024-OKM-4**.

