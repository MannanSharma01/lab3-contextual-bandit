# Lab 3: Contextual Bandit-Based News Article Recommendation System


**Student:** Mannan Sharma  
**Roll Number:** U20230035  
**Course:** Reinforcement Learning Fundamentals  
**Date:** February 2026

---

## Problem Formulation

### Environment Structure
- **Contexts:** 3 unique user types (User1, User2, User3)
- **Bandits/Arms:** 4 news categories per context (Entertainment, Education, Tech, Crime)
- **Total Arms:** 12 (3 contexts × 4 categories)
- **Horizon:** T = 10,000 steps

### Arm Mapping
```
Arms 0-3:     Entertainment, Education, Tech, Crime     → User1
Arms 4-7:     Entertainment, Education, Tech, Crime     → User2
Arms 8-11:    Entertainment, Education, Tech, Crime     → User3
```

---

## Methodology

### 1. Data Preprocessing
- **News Articles Dataset:** Loaded `news_articles.csv` containing articles with category labels
- **User Data:** Split `train_users.csv` and `test_users.csv` for classification
- **Feature Engineering:** Converted all features to numeric format, handled missing values with zero-filling
- **Normalization:** Applied StandardScaler for feature standardization

### 2. User Classification
- **Algorithm:** Random Forest Classifier (100 estimators, max_depth=15)
- **Training:** 80% of `train_users.csv` (1600 samples)
- **Validation:** 20% of `train_users.csv` (400 samples)
- **Performance:** 
  - Validation Accuracy: **85.75%**
  - Precision/Recall per class: ~0.82-0.90
  - This classifier serves as the **context detector** for the bandit system

  - **Why I used Random Forest Classifier & n_estimators=10:**
    - Robust to heterogeneous and noisy features; captures non-linear interactions.
    - Ensemble averaging reduces overfitting and yields stable out-of-the-box performance.
    - `n_estimators=10` chosen to balance training speed and ensemble diversity for the dataset size.
    - Provides simple feature-importance estimates for model interpretability.

### 3. Contextual Bandit Algorithms

#### 3.1 Epsilon-Greedy Strategy
**Algorithm Description:**
- With probability ε: select random arm (exploration)
- With probability (1-ε): select best arm so far (exploitation)
- Maintains count and cumulative reward per arm

**Hyperparameter Tuning:**
- Tested ε values: 0.05, 0.1, 0.2
- **Results:**
  - ε=0.05: Final Avg Reward = **3.6882** (Best for Epsilon-Greedy)
  - ε=0.1:  Final Avg Reward = 3.4670
  - ε=0.2:  Final Avg Reward = 2.9737

**Key Observations:**
- Lower ε values achieve better long-term rewards
- ε=0.05 balances exploration and exploitation effectively
- Higher ε causes excessive exploration, degrading performance

#### 3.2 Upper Confidence Bound (UCB)
**Algorithm Description:**
- Maintains optimistic estimate: avg_reward + C×√(ln(t) / count)
- Exploration bonus decreases over time as uncertainty reduces
- Theoretically principled approach with logarithmic regret bounds

**Hyperparameter Tuning:**
- Tested C values: 0.5, 1.0, 2.0
- **Results:**
  - C=0.5: Final Avg Reward = **3.9281** (Best Overall)
  - C=1.0: Final Avg Reward = 3.9027
  - C=2.0: Final Avg Reward = 3.8842

**Key Observations:**
- **UCB outperforms other algorithms** with best final reward of 3.9281
- Lower C values (0.5) achieve best performance
- Provides stable, smooth convergence trajectory

#### 3.3 SoftMax (Boltzmann Exploration)
**Algorithm Description:**
- Probabilistic arm selection using softmax distribution
- Selects arms proportional to their expected rewards
- Temperature τ controls exploration level

**Hyperparameter Tuning:**
- Tested τ values: 0.5, 1.0, 2.0
- **Results:**
  - τ=0.5: Final Avg Reward = **3.8898** (Best for SoftMax)
  - τ=1.0: Final Avg Reward = 3.8420
  - τ=2.0: Final Avg Reward = 3.2861

**Key Observations:**
- Lower temperature (0.5) performs best
- High temperature (2.0) leads to excessive exploration
- Performance competitive with Epsilon-Greedy

### 4. RL Simulation Methodology (Context-Specific Tracking)
**Simulation Architecture:**
- **Temporal Horizon:** T = 10,000 steps as per assignment specification
- **Test Population:** 2,000 users from test_users.csv
- **Sampling Strategy:** Random user selection at each time step (with replacement)
- **Per-Context Tracking:** Rewards accumulated separately for each of 3 user contexts (user_1, user_2, user_3)

**Key Implementation Details:**
- For each algorithm variant, 3 separate bandits maintained (one per context)
- At each step t:
  1. Randomly sample user from test set
  2. Predict context using trained classifier
  3. Select arm using context-specific bandit
  4. Obtain reward via rlcmab_sampler
  5. Update context-specific bandit with reward
  6. Accumulate reward per context
- Rewards tracked at checkpoints: every 1000 steps (10 checkpoints total)
- Final average reward calculated as: cumulative_reward / total_steps_in_context

**Reason for performing context-specific tracking:**
- This ensures each user type's convergence is independently analyzed
- Reveals context-dependent algorithm performance (may vary per user type)
- Enables identification of best algorithm-hyperparameter-context combination

### 5. Recommendation Engine
**End-to-End Pipeline:**
1. **Classify:** Predict user context using trained Random Forest
2. **Select:** Use trained bandit policy to select optimal category
3. **Recommend:** Randomly sample article from selected category
4. **Transfer:** Seamless integration of classification + RL components

**Features:**
- Supports all three algorithm variants
- Handles feature scaling and missing value imputation
- Returns: user context, recommended category, article headline, URL

---

## Evaluation & Results

### 5.1 Classification Accuracy
```
Classification Report (Validation Set):
              precision    recall  f1-score   support
      user_1       0.85      0.89      0.87       142
      user_2       0.90      0.85      0.87       142
      user_3       0.82      0.83      0.82       116
    accuracy                           0.86       400
```

### 5.2 RL Simulation Results (Per-Context Analysis)
**Simulation Parameters:** T = 10,000 steps, 2,000 test users, Context-Specific Reward Tracking

**Per-Context Final Average Rewards:**

**USER_1 Context Performance:**
```
Algorithm              Hyperparameter         Final Avg Reward
───────────────────────────────────────────────────────────────
Epsilon-Greedy         ε=0.05                    3.4825
Epsilon-Greedy         ε=0.1                     3.1856
Epsilon-Greedy         ε=0.2                     2.5280
UCB                    C=0.5                ***  3.8121 (BEST)
UCB                    C=1.0                     3.7791
UCB                    C=2.0                     3.7891
SoftMax                τ=0.5                     3.8104 (close 2nd)
SoftMax                τ=1.0                     3.7370
SoftMax                τ=2.0                     3.1748
```

**USER_2 Context Performance:**
```
Algorithm              Hyperparameter         Final Avg Reward
───────────────────────────────────────────────────────────────
Epsilon-Greedy         ε=0.05                    2.5189
Epsilon-Greedy         ε=0.1                     2.2897
Epsilon-Greedy         ε=0.2                     1.7549
UCB                    C=0.5                ***  2.7669 (BEST)
UCB                    C=1.0                     2.7348
UCB                    C=2.0                     2.7317
SoftMax                τ=0.5                     2.7178 (close 2nd)
SoftMax                τ=1.0                     2.7221
SoftMax                τ=2.0                     2.0344
```

**USER_3 Context Performance:**
```
Algorithm              Hyperparameter         Final Avg Reward
───────────────────────────────────────────────────────────────
Epsilon-Greedy         ε=0.05                    5.4814
Epsilon-Greedy         ε=0.1                     5.3355
Epsilon-Greedy         ε=0.2                     5.1327
UCB                    C=0.5                     5.5961
UCB                    C=1.0                ***  5.6079 (BEST)
UCB                    C=2.0                     5.5748
SoftMax                τ=0.5                     5.4730
SoftMax                τ=1.0                     5.4697
SoftMax                τ=2.0                     5.1328
```

**Key Observations:**
- **USER_1:** UCB (C=0.5) optimal, moderate reward range (3.1-3.8)
- **USER_2:** UCB (C=0.5) optimal, lowest reward range (1.7-2.7) - suggests harder problem
- **USER_3:** UCB (C=1.0) optimal, highest reward range (5.1-5.6) - rich reward structure
- **Consistency:** UCB family dominates for best hyperparameter across all contexts

### 5.3 Analysis Plots (Context-Specific Analysis)
Two comprehensive visualizations have been generated as per assignment requirements:

1. **reward_vs_time_per_context.png** **PRIMARY PLOT**
   - **REQUIREMENTS COMPLIANCE:** Shows "Average Reward vs. Time FOR EACH CONTEXT" per assignment spec
   - Three subplots: USER_1, USER_2, USER_3 (each showing all 9 algorithm variants)
   - Each context displays convergence curves for:
     - Epsilon-Greedy: ε=0.05, 0.1, 0.2 (red shades)
     - UCB: C=0.5, 1.0, 2.0 (teal shades)
     - SoftMax: τ=0.5, 1.0, 2.0 (yellow shades)
   - Context-specific observations:
     - **USER_1:** Rewards converge to ~3.1-3.8 range; UCB/SoftMax (C=0.5,τ=0.5) best
     - **USER_2:** More variable rewards (~1.7-2.7); different arms per context
     - **USER_3:** Highest rewards (~5.1-5.6); distinct per-context characteristics visible

2. **hyperparameter_comparison_per_context.png**
   - Bar chart with context breakdown (three colors: user_1, user_2, user_3)
   - Three subplots: Epsilon-Greedy, UCB, SoftMax algorithms
   - Shows final average reward for each context within each algorithm
   - Clearly demonstrates context-dependent performance differences

---

## Key Findings & Insights

### 1. Context-Specific Performance (NEW)
Per-context analysis reveals significant performance differences across user types:

**USER_1 Performance:**
- UCB (C=0.5): 3.8121 *** Best for this context
- SoftMax (τ=0.5): 3.8104 - Nearly equal to UCB
- Epsilon-Greedy (ε=0.05): 3.4825 - Competitive but lower
- **Insight:** User_1 benefits from structured exploration (UCB) over random exploration

**USER_2 Performance:**
- UCB (C=0.5): 2.7669 *** Best for this context
- SoftMax (τ=0.5): 2.7178 - Comparable
- Epsilon-Greedy (ε=0.05): 2.5189 - Notably lower
- **Insight:** User_2 context shows hardest reward structure (lowest absolute rewards)

**USER_3 Performance:**
- UCB (C=1.0): 5.6079 *** Best for this context
- UCB (C=0.5): 5.5961 - Equally effective
- SoftMax (τ=0.5): 5.4730 - Close second
- **Insight:** User_3 context has richest rewards; UCB with moderate C works best

### 2. Algorithm Performance Ranking (Overall)
1. **UCB (C=0.5):** Best across all contexts with consistent high performance
2. **SoftMax (τ=0.5):** Highly competitive, strong robustness
3. **Epsilon-Greedy (ε=0.05):** Solid performer but context-dependent

### 3. Hyperparameter Sensitivity
- **Lower exploration parameters perform better** consistently
- Context-specific optimal hyperparameters vary:
  - User_3 prefers UCB with C=1.0 over C=0.5
  - Users 1-2 prefer lower C values
- **Dynamic hyperparameter tuning per context could improve results**

### 4. Context-Dependent Characteristics
- **User_1:** Middle reward range (3.1-3.8), balanced exploration needs
- **User_2:** Lower reward range (1.7-2.7), possibly harder distinctions between arms
- **User_3:** Higher reward range (5.1-5.6), more separable reward structures
- **Implication:** Article choices strongly depend on user context preferences

### 5. Convergence Trajectories
- **UCB:** Smooth, principled convergence across all contexts
- **SoftMax:** Stable probabilistic learning, minimal oscillation
- **ε-Greedy:** Fast initial gains, context-dependent convergence speed
- **All algorithms:** Clear convergence within first 2000-3000 steps

---

## How to Reproduce

### Requirements
```bash
pip install rlcmab-sampler pandas numpy scikit-learn matplotlib
```

### Execution Steps
1. **Open notebook:** `lab3_results_U20230035.ipynb` in Jupyter/VS Code
2. **Configure environment:** Python 3.12+ with required packages
3. **Run cells sequentially:**
   - Cells 1-4: Data loading and preprocessing
   - Cells 5-6: User classifier training and validation
   - Cells 7-11: Sampler, arm mapping, algorithm implementations
   - Cell 12: RL simulation (T=10,000) - **Takes 2-3 minutes**
   - Cell 13: Generate plots
   - Cell 14: Recommendation engine examples

### Output Files Generated
- `lab3_results_U20230035.ipynb` - Main notebook with all code and results
- `reward_vs_time_per_context.png` - **PRIMARY:** Context-specific convergence plots (3 contexts × 9 algorithm variants)
- `hyperparameter_comparison_per_context.png` - Final reward comparison with context breakdown
- `README.md` - This documentation

---

## Detailed Implementation Notes

### Sampler Integration
The `rlcmab_sampler` package is initialized with roll number **35** (from U20230035):
```python
from rlcmab_sampler import sampler
reward_sampler = sampler(35)
reward = reward_sampler.sample(arm_index)  # arm_index: 0-11
```

### Feature Engineering
- **Train Users:** 2000 samples with 30+ features
- **Test Users:** 2000 samples (no labels, for evaluation)
- **Missing Value Handling:** Filled with zeros (common for sparse features)
- **Feature Scaling:** StandardScaler fit on training set, applied to validation/test

### Bandit Training Details
- **Architecture:** Separate bandits per context (3 × 3 algorithms × 3 hyperparameters = 27 total bandits)
  - User_1 Bandits: EG-ε0.05, EG-ε0.1, EG-ε0.2, UCB-C0.5, UCB-C1.0, UCB-C2.0, SM-τ0.5, SM-τ1.0, SM-τ2.0
  - User_2 Bandits: (Same 9 variants)
  - User_3 Bandits: (Same 9 variants)
- **Online Learning:** Rewards obtained instantly via sampler.sample(arm_index)
- **Stochastic Sampling:** Random user selection from 2000 test users at each timestep
- **Per-Context Reward Accumulation:** 
  - Each context maintains its own cumulative reward counter
  - At each step t: determines user → looks up context → selects arm from context-specific bandit → gets reward → updates context-specific counter
  - Enables calculation of final average reward per context
- **No Batch Updates:** Pure online RL approach with immediate reward feedback

### Reproducibility
- Random seed fixed: `np.random.seed(42)`
- Consistent train/test split: `random_state=42` in `train_test_split`
- Results should be reproducible across runs