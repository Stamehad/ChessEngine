import torch
import pytorch_lightning as pl
from chessengine.model.engine_pl import ChessLightningModule
from chessengine.chess_rl.self_play.initial_state_sampler import InitialStateSampler
from chessengine.chess_rl.self_play.beam_game_simulator import BeamGameSimulator
from dotenv import load_dotenv
import os
import yaml


class ChessSelfPlayTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        load_dotenv()
        CHECKPOINT_PATH = os.getenv("BASE_MODEL")

        self.model = ChessLightningModule.load_from_checkpoint(CHECKPOINT_PATH, config=config)
        self.save_hyperparameters()

        self.sampler = InitialStateSampler(config=config)
        self.game_sim = BeamGameSimulator(model=self.model, config=config)

        self.batch_buffer = []  # <--- Important! Store batches here
        self.batch_size = 512   # Set your desired training batch size
        self.automatic_optimization = False

    def forward(self, x, legal_moves):
        return self.model(x, legal_moves)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=your_learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        # ===> 1. If no batch left, generate new batches
        if len(self.batch_buffer) == 0:
            boards = self.sampler.get_boards()
            data = self.game_sim.play_games(boards, n_games=40)

            self.batch_buffer = self.split_data_into_batches(data)

        # ===> 2. Get one batch
        batch = self.batch_buffer.pop()

        # ===> 3. Usual forward/loss/backward/step
        x = batch['board']
        legal_moves = batch['labels']['legal_move']
        true_index = batch['labels']['true_index']
        eval_label = batch['labels']['eval']

        x_out, move_pred = self.model(x)

        move_loss = compute_move_loss(move_pred, legal_moves, true_index)
        eval_loss = compute_eval_loss(x_out, eval_label)

        loss = move_loss + eval_loss

        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        self.log('loss', loss)
        self.log('move_loss', move_loss)
        self.log('eval_loss', eval_loss)

        return loss

    def split_data_into_batches(self, data):
        """
        Split the output of play_games into small batches for training.
        """
        x, legal_moves, evals, true_indices = data  # each is (N, ...)

        N = x.shape[0]
        batches = []

        for start in range(0, N, self.batch_size):
            end = start + self.batch_size
            batch = {
                'board': x[start:end],
                'labels': {
                    'legal_move': legal_moves[start:end],
                    'true_index': true_indices[start:end],
                    'eval': evals[start:end],
                }
            }
            batches.append(batch)

        return batches