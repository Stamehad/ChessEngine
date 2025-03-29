# Chess Engine Transformer (Self-Supervised + RL)

This project implements a transformer-based chess engine designed to learn in a **self-supervised** manner and eventually integrate **reinforcement learning**. It deviates from traditional approaches in several key ways to support scalability, flexibility, and interpretability.

---

## ✨ Core Design Principles

### 1. 🧠 Representation over Policy
- Instead of directly predicting moves (e.g., with a 64 × 64 × N move head), the model learns **rich representations** of board positions.
- This avoids:
  - Predicting illegal moves
  - Handling complex move structures (e.g. promotions, castling) symbolically
- Focus shifts to modeling **state evolution and evaluation**.

---

### 2. 🎯 Move Prediction Without a Move Head

Unlike traditional engines that predict moves using a massive **(from_square × to_square × promotion_type)** classifier, this model **avoids predicting symbolic moves directly**.

Instead:

- The model is trained to predict the **correct piece (or empty)** on **squares that change** between consecutive positions.
- This naturally captures:
  - Standard moves (2 changed squares)
  - Promotions (changed square has a different piece)
  - En passant (2 changed squares, including captured pawn)
  - Castling (4 changed squares)

> This approach removes the need to model illegal moves entirely — it always operates on real board positions.

---

### 3. ♻️ Recycling and Iterative Refinement
- Inspired by AlphaFold2, the model applies its transformer stack **multiple times per input**.
- Each application of the model is not a move simulation but a **refinement** of the current latent representation.

> The model is not a simulator; it is a reasoner.

- Recycling enables:
  - Improved depth of reasoning without growing model depth
  - Time vs. compute tradeoffs during inference
  - Training stability through shallow supervision

---

### 4. 🏗 Refined Transformer Architecture

```python
# High-level pseudocode for inference:
x = one_hot_encoded_board()
X0 = BoardEmbedding(x)
Y0 = Transformer(X0)

for i in range(N - 1):
    delta = RecycleEmbed(Yi.detach())
    Yi+1 = Transformer(X0 + delta)

eval_score = EvalHead(YN)
piece_logits = MoveHead(YN)  # Predict piece identity for each square (used for changed-square loss or move scoring)
```

### Components
- **BoardEmbedding**: Maps one-hot board input to square-wise vectors
- **Transformer**: Stack of attention blocks with layer norm, dropout, SwiGLU
- **RecycleEmbed**: Processes transformer output to produce a residual update
- **EvalHead**: Predicts evaluation score in [0, 1] representing win probability for white
- **MoveHead**: Predicts the correct piece (or empty) on squares that changed from one position to the next

---

### 5. 🔁 Stochastic Recycling Schedule
- During training, the number of recycling steps is sampled from `[1, 2, 3, 4]` (or configurable).
- Ensures the model can:
  - Perform well under variable compute budgets
  - Learn meaningful refinements at every step (not just final)
- At inference, maximum number of recycles is always used.

---
### 6. 🧪 Self-Supervised Training Objective

The model is trained without labeled moves or engines — instead, it uses self-supervision derived from raw game data.

#### 🏁 Current Loss: Evaluation Loss
- Predicts the win probability for white from a given board position.
- **Target**: game result (1 = white win, 0.5 = draw, 0 = black win)
- **Loss**: Mean Squared Error (MSE) between predicted and target evaluation.

#### 🧠 Move Supervision via Changed-Square Prediction
Instead of predicting moves directly, the model learns to **predict what changes** between two positions.

For each consecutive board pair \((X_i, X_{i+1})\):
- Detect which squares changed.
- For each changed square:
  - Extract the vector from the model output \(Y_i = f(X_i)\)
  - Project it through a small head into logits over pieces (including empty)
  - Apply cross-entropy loss against the piece that appears in \(X_{i+1}\)

This approach captures:
- All legal move types (normal, promotion, en passant, castling)
- Only real, legal transitions
- Minimal output space — no giant 64×64 move matrices

#### 🧪 Future Self-Supervised Losses
- Contrastive prediction of next states
- Denoising / masked square recovery
- RL-based fine-tuning using evaluation and move outcomes

---

### 🤖 Inference-Time Move Selection

At inference time:
- Generate all legal moves from a position
- For each move:
  - Simulate the resulting board
  - Identify changed squares
  - Use the model to score how likely the new pieces are on those squares
- Pick the move with the **highest total log-probability**

---

## 🚧 Next Steps
- [ ] Implement move scoring by evaluating changed squares under all legal moves
- [ ] Add intermediate supervision on recycled steps (optional)
- [ ] Explore RL fine-tuning using evaluation score + move selection

---

## 🧩 Design Summary

| Feature                 | Motivation                                                 |
|------------------------|------------------------------------------------------------|
| Attention Transformer  | Global reasoning across the whole board                    |
| Learned Positional Embedding | Let model learn square-specific inductive biases     |
| Recycling              | Depth-time tradeoff, iterative reasoning                   |
| Evaluation Head Only   | Avoids sparse move space; useful for RL                    |
| Self-Supervised Loss   | Learn from raw games without expert labels                 |
| Residual Refinement    | Better gradient flow, inspired by AlphaFold2               |

---

### 7. 📚 Embedded Input Features

The input tensor for each position has shape `(8, 8, 20)`. The 20 feature channels include:

| Channel | Description                                            |
|---------|--------------------------------------------------------|
| 0       | Empty square (1 if empty, else 0)                      |
| 1–6     | White pawn, knight, bishop, rook, queen, king          |
| 7–12    | Black pawn, knight, bishop, rook, queen, king          |
| 13      | Side to move (1 = white to move, 0 = black)            |
| 14      | In-check flag for the player's king (broadcasted to all squares) |
| 15      | Threatened flag — this piece is under threat (regardless of color) |
| 16      | Threatening flag — this piece threatens opponent pieces (player-relative) |
| 17      | Legal move — this piece has at least one legal move    |
| 18      | Player has castling rights (1 if yes, 0 if not)        |
| 19      | Opponent has castling rights (1 if yes, 0 if not)      |

These features are used by the `BoardEmbedding` module to generate per-square input embeddings of shape `(64, H)`.

---

### 8. 📦 Batch Format

Each training batch consists of a dictionary with the following entries:

| Key              | Shape      | Description                                                  |
|------------------|------------|--------------------------------------------------------------|
| `board`          | (8, 8, 20) | The raw input tensor for a position                         |
| `eval`           | (1,)       | Scalar win probability target (1=white win, 0=black win)     |
| `move_target`    | (64,)      | Labels for changed squares (0=empty, 1–6=piece type)         |
| `king_square`    | (1,)       | Index [0–63] of opponent king square                         |
| `check`          | (1,)       | Whether opponent king is in check (0 or 1)                   |
| `threat_target`  | (64,)      | Labels for newly threatened opponent pieces (0 or 1)         |
| `threat_mask`    | (64,)      | Mask of which squares are valid for threat prediction        |
| `move_weight`    | (1,)       | Optional per-sample weighting for move prediction loss       |

These tensors are used to compute the various loss components during training.