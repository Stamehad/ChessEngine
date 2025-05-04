from chessengine.model.engine_pl import ChessLightningModule
from chessengine.chess_rl.mcts.mcts import BATCHED_MCTS
import chess
import chess.pgn
import torch
import yaml
import io
import json
import os
from dotenv import load_dotenv
load_dotenv()
CHECKPOINT_PATH = os.getenv("BASE_MODEL")


# load configs
with open("engine_config.yaml", "r") as f:
    config = yaml.safe_load(f)

rl_config = {
    'mcts':{"num_simulations": 1, "cpuct": 1.0, "temperature_start": 1.0,}
}

# load model and mcts
model = ChessLightningModule.load_from_checkpoint(CHECKPOINT_PATH, config=config)
model.eval()
mcts = BATCHED_MCTS(model, rl_config)

# load some random boards
pgn_file = "data/shards300_small/shard_0.pgn"
# read the first 10 games from the pgn file
games = {}
with open(pgn_file, 'r') as f:
    for j, line in enumerate(f):
        obj = json.loads(line)
        pgn = obj["pgn"]

        games[f'game{j}'] = chess.pgn.read_game(io.StringIO(pgn))
        if j == 100:
            break

boards = {}
for i, game in enumerate(games.values()):
    # get random number of moves
    num_moves = torch.randint(4, 20, (1,)).item()
    board = game.board()
    for j, move in enumerate(game.mainline_moves()):
        board.push(move)
        if j == num_moves:
            break

    boards[f'board{i}'] = board

boards = [board for board in boards.values()]

########## TEST MCTS RUNTIME ##########
# import time
# for n in range(10, 110, 10):
#     rl_config["mcts"]["num_simulations"] = n
#     mcts = BATCHED_MCTS(model, rl_config)
#     # Run MCTS on the boards
#     start_time = time.time()
#     pis = mcts.run_mcts_search(boards) # [(legal_moves, pi), ...]
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Elapsed time for {n} simulations: {elapsed_time:.2f} seconds")
    

# Run MCTS on the boards
for n in range(10, 110, 10):
    print(f"Running MCTS with {n} boards in parallel...")
    bs = boards[:n]
    pis = mcts.run_mcts_search(boards) # [(legal_moves, pi), ...]

# print(pis[0])

# # Check the output
# assert len(pis) == len(boards)
# for legal_moves, pi in pis:
#     assert len(legal_moves) > 0
#     assert len(pi) > 0
#     assert len(pi) == len(legal_moves)
#     assert torch.all(pi >= 0)
#     assert torch.all(pi <= 1)
#     assert torch.isclose(pi.sum(), torch.tensor(1.0), atol=1e-5)

def test_mcts():
    """
    Test the MCTS implementation.
    """
    # Run MCTS on the boards
    pis = mcts.run_mcts_search(boards) # [(legal_moves, pi), ...]

    # Check the output
    assert len(pis) == len(boards)
    for legal_moves, pi in pis:
        assert len(legal_moves) > 0
        assert len(pi) > 0
        assert len(pi) == len(legal_moves)
        assert torch.all(pi >= 0)
        assert torch.all(pi <= 1)
        assert torch.isclose(pi.sum(), torch.tensor(1.0), atol=1e-5)