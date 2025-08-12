# Reinforcement Learning Method for Chess Engine Training

## Overview

We start from a **base model** that has already undergone extensive **self-supervised training** on a large database of human games.  
Through this process, the model has developed substantial chess understanding — for instance, it assigns >70% cumulative probability to its top-3 move predictions on average.

However, to reach **high-level move prediction and decision-making**, an engine (like a human player) must **reason beyond static evaluations** and **analyze future variations**.  
To achieve this, we now introduce **reinforcement learning (RL)** methods — allowing the model to **play games against itself**, explore future possibilities, and improve from its own self-play data.

Specifically, we considered two main approaches for search-based RL training:
- **Monte Carlo Tree Search (MCTS)** (as used by AlphaZero)
- **Beam Search** (a more parallelizable alternative)

A detailed comparison follows below.

---

## Training Goals and Constraints

Before diving into method selection, it is important to understand the main **practical constraints** and **training goals**:

- 🖥️ **Hardware:**  
  - A single **A100 GPU**, capable of handling **batch sizes of ~512 positions** without memory issues.
  - In infrence mode can handle over **10K positions**, each model call takes **0.03 seconds**.
  
- 📚 **Data Size Target:**  
  - Base training used **~2M positions**.  
  - For RL we aim for a similar scale:  **~1M new self-play positions**.

- ⏱️ **Training Time:**  
  Aim for **5 hours** (no more than 10).

➡️ It follows that we need around **2K training steps** (~1M/512), each one taking about **10 seconds** (~5Hr/2K).

Naively, we can schuedule 512 games in parallel and get enough positions for a training step with each move. However, the game must end before making a training step because we use the result of the game as a reward for RL. 

- 🔁 **Training Cycle Definition:**  
  We define a **training cycle** as:

  > 1. **Self-play:** Generate 512 × s positions (playing games in parallel) 
  > 2. **Training:** Perform **s training steps**

A single game includes about 100-200 positions (engine games tend to be longer than human games) which comes to around 3-6 second just for model calls. But this is **huge underestimate** as we did not consider analyzing variations. Typically we will want to consider an order of hundreds of positions for each move, i.e. hundrends of model calls. It is crucial to make these calls in parallel as much as possible. 

When analyzing variation it is important to consider the depth level. If we want to look 8 half-moves ahead, then we need at least 8 model calls as different depth levels must be processed sequetially! 


- 🎯 **Self-Play Quality Matters:**  
  Unlike static supervised learning, **self-play data is dynamic**:  
  The **model’s own quality** at a given moment determines the **quality of new training data**, which in turn affects future improvements.  
  This creates a **positive feedback loop**, but also means:

  - Smaller **s** (fewer steps per self-play cycle) enables faster model updates and **better responsiveness**.
  - Larger **s** might introduce **more stability** but risks **lagging behind** if the model improves quickly.

- **Main Challenges:**
  > 1. Variation search that meets the constraints given a bottleneck of 0.03 second for each model call.
  > 2. Dealing with chess logic (i.e. computing legal moves and features tensors) for 10K boards in parallel (in we want to fully saturate the GPU abilities) creates significant overhead if done naively (with python Chess package it completely dominates over the model calls). 

---

## Two Alternatives: MCTS vs Beam Search

### 🌲 Monte Carlo Tree Search (MCTS)

MCTS builds a **search tree** by iteratively:
- **Simulating games** from the current state.
- **Evaluating leaf nodes** using the model's policy and value heads.
- **Backpropagating** the resulting evaluations to guide future searches.

✅ Pros:  
- Very **strong** search quality with few evaluations.  
- Naturally **balances exploration vs exploitation**.  

❌ Cons:  
- **Sequential**: each simulation depends on updated tree statistics.
- Harder to **parallelize** inference.
- Model needs to be called **100× or more** sequentially per move!

---

### 🌟 Beam Search

Beam Search instead expands a **fixed-width tree**:
- At each layer, it **keeps top-k moves** (by model probability or evaluation).
- Expands **all** current nodes **in parallel**.
- Can reach significant depth while making only a **small, bounded number of model calls**.

✅ Pros:  
- Highly **parallelizable** on GPU.
- **Batched** inference at every layer.
- More efficient for large-scale position generation.

❌ Cons:  
- Less intelligent search compared to full MCTS.
- Cannot dynamically adjust exploration during search.

---



## Main Trade-offs and Final Choice

| Dimension | MCTS | Beam Search |
|:---|:---|:---|
| Inference mode | Sequential (hard to batch) | Fully parallel |
| Per-move model calls | ~100 | ~5–8 |
| Search quality | Higher | Good enough |
| Games per cycle | Few | Many |
| Position diversity | Low (more deterministic) | Higher (naturally branched) |

Therefore we use **Beam Search** for now.

**Key reasons:**
- **Parallelization:** It is critical to maximize GPU utilization during data generation.
- **Throughput:** Beam Search allows generating **many more positions** per unit time, enabling faster RL training.
- **Stability:** Wider branching and multiple games in parallel naturally increase **diversity** of positions without heavy exploration tuning.

--- 

## Beam Search Details

1. Use Beam Search with a **top-k schedule** [8,5,3,2,1,1,1]. This means starting from root position we first choose the best 8 moves, then for each we choose the best 5 moves, then 3 and so on. We get the expansion: 1 -> 8 -> 40 -> 120 -> 240 -> 240 -> 240 -> 240, which means that for each root position we evaluate 1128 positions.

2. Now suppose we want to have N games in parallel. With a maximal batch size of around 10K we are limited to about 40 parallel games on maximal expansion. But that means the batch size increases from 40 to 9600 during expansion which means the GPU is not used to its full capacity each time the model is called. To rectify this we schedule the games in the following way:

<pre>
```
1
8,   1
40,  8,   1
120, 40,  8,   1 
240, 120, 40,  8,   1
240, 240, 120, 40,  8,   1
240, 240, 240, 120, 40,  8,   1
240, 240, 240, 240, 120, 40,  8,   1
1,   240, 240, 240, 240, 120, 40,  8
8,   1,   240, 240, 240, 240, 120, 40
40,  8,   1,   240, 240, 240, 240, 120
120, 40,  8,   1,   240, 240, 240, 240
240, 120, 40,  8,   1,   240, 240, 240
240, 240, 120, 40,  8,   1,   240, 240
240, 240, 240, 120, 40,  8,   1,   240
```
</pre>


This way for a block of 8 games we get to a fix batch size of 1129. Duplicating this structure 8 times we get 64 games in parallel with a fixed batch size of 9,032. 

3. For 100 positions a game (at least) we get 6400 positions per training cycle and s=12 training steps. Moreover, ~800 model calls to finish the games, which is 24 seconds just for the model run. This s is too big (too many steps for the same model quality) and the game cycle is a bit too long. We can use the following trick. For each root position we search we get a principal variation (PV), a list of best moves starting from the root position. At least the first few ones are quite high quality so instead of only pushing the first move in the PV we can push 3 moves (so that the color alternates). This way we only perform a full Beam Search for a third of the positions in a game, leading to s=4 training steps with 8 seconds total model run. 

4. Sampling initial positions: so that the 64 games are not copies of each other (the model evaluation and beam search is completely deterministic) and also to get a good diversity of opennings we sample from:
    - Black initial positions after white played one of its 20 possible openinig moves.
    - White initial positions after black played its first move (400 possible positions).
    - Random positions from the online games used in based training.

---

