import yaml
import torch, chess, random
from chessengine.chess_rl.beam_search.beam_tree import BeamPosition, BeamLayer, TrainingBeamTree
from chessengine.model.engine_pl import ChessLightningModule
import os 
from dotenv import load_dotenv

with open("engine_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

load_dotenv()
CHECKPOINT_PATH = os.getenv("BASE_MODEL")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
config['train']['device'] = device

model = ChessLightningModule(config=config)
model = ChessLightningModule.load_from_checkpoint(CHECKPOINT_PATH, config=config)
model.eval()
model.to(device)



class DummyModel(torch.nn.Module):
    def forward(self, x):
        B = x.shape[0]
        # fake “value” head (B,65,H) – we only need one vector to call get_eval_prob
        fake_val = torch.zeros(B, 65, 1)
        # fake “policy” head  (B,64,7) – uniform noise
        fake_pol = torch.randn(B, 64, 7)
        return fake_val, fake_pol
    
def random_board(max_moves=20):
    board = chess.Board()
    for _ in range(random.randint(0, max_moves)):
        if board.is_game_over(): break
        board.push(random.choice(list(board.legal_moves)))
    return board

def test_backprop_cascade():
    
    tree  = TrainingBeamTree(model, 
                             device=device,
                             topk_schedule={0:2, 1:1},  # super-shallow
                             pv_skip=1)
    tree.setup([random_board() for _ in range(4)], n_packets=2)
    tree.play_until_done()

    # no active roots or packets left
    assert not tree.active_roots
    assert not tree.packet_queue

    # every training record is a tuple (fen, move, result)
    for fen, mv, res in tree.training_records:
        assert res in (0,1,2)
        bd = chess.Board(fen)
        if mv is not None:
            assert chess.Move.from_uci(mv) in bd.legal_moves

def test_no_memory_leak():
    import gc, psutil, os
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss

    tree  = TrainingBeamTree(model, 
                             device=device,
                             topk_schedule={0:2, 1:1},  # super-shallow
                             pv_skip=1)
    tree.setup([random_board() for _ in range(16)], n_packets=4)
    tree.play_until_done()

    rss1 = proc.memory_info().rss
    # allow 30 MB drift
    assert rss1 - rss0 < 30 * 1024 * 1024


# run the test
if __name__ == "__main__":
    test_no_memory_leak()
    print("Test passed!")

if __name__ == "__main__":
    import time
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['train']['device'] = device.type

    model = ChessLightningModule(config=config)
    model = ChessLightningModule.load_from_checkpoint(CHECKPOINT_PATH, config=config)
    model.eval()
    model.to(device)

    tree = TrainingBeamTree(model,
                            device=device,
                            topk_schedule={0: 2, 1: 1},
                            pv_skip=1)
    boards = [random_board() for _ in range(8)]
    tree.setup(boards, n_packets=4)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
    else:
        mem_before = None

    start_time = time.time()
    tree.play_until_done()
    end_time = time.time()

    if device.type == 'cuda':
        mem_after = torch.cuda.memory_allocated()
        mem_peak = torch.cuda.max_memory_allocated()
    else:
        mem_after = None
        mem_peak = None

    print(f"Full game cycle on device {device} took {end_time - start_time:.3f} seconds.")
    if device.type == 'cuda':
        print(f"CUDA memory allocated before: {mem_before} bytes")
        print(f"CUDA memory allocated after: {mem_after} bytes")
        print(f"CUDA peak memory allocated during run: {mem_peak} bytes")
