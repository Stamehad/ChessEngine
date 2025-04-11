# Chess Engine Transformer (Self-Supervised + RL)

This project implements a transformer-based chess engine designed to learn in a **self-supervised** manner and eventually integrate **reinforcement learning**. 

---

## ⚙️ Core Design Overview

- Uses a Transformer as the core architecture for global board reasoning
- Trained first via self-supervised objectives, then refined via reinforcement learning
- Predicts local changes in board state (square-level changes) instead of symbolic moves
- Incorporates recycling (iterative refinement of internal state)
- Evaluates positions via a scalar win probability score

---

## 📊 Data

- Source: Lichess games database, February 2025
- Filtered for games with:
  - Both players rated above 2400 Elo
  - Classical or rapid time controls (>5 minutes)
- Final dataset:
  - 317,083 games
  - 2,370,238 total positions used for training

---

## 🔄 Parsing and Input Encoding

The board positions are parsed into tensors of shape `(8, 8, 21)`.

- **Channels 0–12** represent piece identity:
  - 0: empty square
  - 1–6: white pawn, knight, bishop, rook, queen, king
  - 7–12: black pawn, knight, bishop, rook, queen, king

See the full feature table at the end for the remaining 13–20 feature channels.

---

## 🧠 Architecture Overview

The model refines its understanding of the board by recycling its transformer output:

```python
# High-level pseudocode
P                                 # Game position 
E = one_hot_encoded_board(P)      # (8,8,21) board tensor
X0 = BoardEmbedding(E)            # (64, H)
Y0 = Transformer(X0)

for i in range(N - 1):
    delta = RecycleEmbed(Yi.detach())
    Yi+1 = Transformer(X0 + delta)

#Prediction heads
eval_score = EvalHead(YN)         # (1,) Eval prediction
piece_logits = MoveHead(YN)       # (64, 7) Predict piece at each square 
in_check_logit = InCheckLoss(YN)  # (1,) Is the opponent in check after the move
in_threat_logits = ThreatHead(YN) # (64, 2) Threat after move prediction per square
```

- **BoardEmbedding** maps board tensor into (64, H) embeddings
- **Transformer** stack models position with global attention
- **RecycleEmbed** refines the representation with residual feedback
- **EvalHead** predicts the probability white will win
- **MoveHead** predicts the updated pieces at changed squares

### 🔧 Transformer Hyperparameters

- Embedding dimension: 128
- Number of attention heads: 8
- Number of layers: 12
- MLP hidden dimension: 128 × 4 (via `mlp_ratio = 4`)
- Dropout: 0.1

🔁 Stochastic Recycling Schedule

- During training, the number of recycling steps is sampled from [1, 2, 3, 4] (or configurable)
- At inference, maximum number of recycles is always used
---

## 🎯 Losses and Training

- **Evaluation Loss**: Predict scalar win probability ∈ [0, 1] for each position.
  - Loss: MSE against actual game result (1 = white win, 0.5 = draw, 0 = black win)

- **Move Supervision (Changed-Square Loss)**:
  - The MoveHead outputs 7 logits per square (0 = empty, 1–6 = piece types), resulting in a `(64, 7)` prediction.
  - For each position, we collect all legal moves into a tensor `legal_moves (64, L)`, which identifies:
    - The set of **squares that change** under each legal move
    - The **correct piece** that should occupy each changed square (including empty)
  - This structure allows handling complex moves:
    - Normal moves affect 2 squares (from and to)
    - Castling affects 4 squares (king and rook)
    - En passant affects 3 squares (pawn from, pawn to, captured pawn)
  - For each of L legal moves the logits of changed squares are averaged to a single logits
  - The loss is a cross-entropy between predicted legal moves probabilities and ground truth move. 

- **Additional Auxiliary Losses:**
  - Threat prediction loss: Predicts which squares become threatened (for both player and opponent) as a result of the move
  - In check loss: Predicts whether the opponent is in check after the move

- Future planned objectives:
  - Contrastive state prediction
  - Denoising / masked square prediction
  - RL fine-tuning based on move outcomes and eval score

---

## 📏 Metrics

To evaluate training progress and model quality, we track the following metrics:

- **Move Accuracy**: Percentage of positions where the highest-scoring move (based on changed-square logits) matches the ground truth move played in the game.
- **Correct Move Probability**: Average probability assigned to the ground truth move across the dataset, measuring model confidence in the correct choice.
- **Top-k Move Coverage**: For `k` in {1, 3, 5}, the average total probability mass assigned to the top-k scoring moves, showing how concentrated the model's predictions are.
- **Low Probability Fraction**: For positions with at least 20 legal moves, measures the average fraction of legal moves assigned less than 1% probability. This indicates how sparse or confident the model's distribution is.
- **Top-k Accuracy**: For `k` in {3, 5}, the probability that the ground truth move is in top-3/5 predicted moves. 

---

## 🧠 Inference-Time Move Selection

To select a move during inference:

- Use the `MoveHead` output `(64, 7)` to get logits for each square and possible piece.
- For each legal move (from the `legal_moves` tensor):
  - Identify the set of changed squares and the expected piece at each.
  - Compute the average log-probability of those predicted pieces using `MoveHead` outputs.
- Assign each move a total score based on the average log-probability across changed squares.
- Select the move with the highest score.

This is a zero-order estimate (no rollout or simulation), relying entirely on the model’s current belief about likely state transitions.

---

### 7. 📚 Embedded Input Features

The input tensor for each position has shape `(8, 8, 21)`. The 21 feature channels include:

| Channel | Description                                                              |
|---------|--------------------------------------------------------------------------|
| 0       | Empty square (1 if empty, else 0)                                        |
| 1–6     | White pawn, knight, bishop, rook, queen, king                            |
| 7–12    | Black pawn, knight, bishop, rook, queen, king                            |
| 13      | Side to move (1 = white to move, 0 = black)                              |
| 14      | In-check flag for the player's king (broadcasted to all squares)         |
| 15      | Threatened flag — this piece is under threat (regardless of color)       |
| 16      | Threatening flag — this piece threatens opponent pieces (player-relative)|
| 17      | Legal move — this piece has at least one legal move                      |
| 18      | White has castling rights (1 if yes, 0 if not)                           |
| 19      | Black has castling rights (1 if yes, 0 if not)                           |
| 20      | En passant square (1 at en passant target square, else 0)                |

These features are used by the `BoardEmbedding` module to generate per-square input embeddings of shape `(64, H)`.

---

### 8. 📦 Batch Format

Each training batch consists of a dictionary with the following entries:

| Key              | Shape      | Description                                                  |
|------------------|------------|--------------------------------------------------------------|
| `board`          | (8, 8, 21) | The raw input tensor for a position                          |
| `eval`           | (1,)       | Scalar win target (2=white win, 1= draw, 0=black win)        |
| `move_target`    | (64,)      | Labels for changed squares (0=empty, 1–6=piece type)         |
| `king_square`    | (1,)       | Index [0–63] of opponent king square                         |
| `check`          | (1,)       | Whether opponent king is in check (0 or 1)                   |
| `threat_target`  | (64,)      | Labels for newly threatened opponent pieces (-100 or 0 or 1) |
| `terminal_flag`  | (1,)       | Game state (0=active, 1=stalemate, 2=checkmate)              |
| `legal_moves`    | (64, L)    | Like move_target (=ground truth) but for all L legal moves   |
| `true_index`     | (1,)       | Index of the ground truth move in legal moves                |