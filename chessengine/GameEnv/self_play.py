import pygame
import chess
import chess.svg
import sys
import numpy as np
from typing import List, Tuple, Optional, Any # Use Any for the custom LegalMove format initially
import torch
from dotenv import load_dotenv
import os
import yaml
from dataclasses import dataclass

from chessengine.preprocessing.position_parsing import encode_board, generate_legal_move_tensor
from chessengine.model.engine_pl import ChessLightningModule
from chessengine.chess_rl.mcts.mcts import BATCHED_MCTS
from chessengine.chess_rl.beam_search.beam import BEAM
from chessengine.model.prediction import batch_predict

class Position:
    def __init__(self, board, pred, idx):
        self.board = board
        self.idx = idx
        probs, evals, legal_move_lists = pred
        self.probs = probs.squeeze(0)  # (L,)
        self.evals = evals.squeeze(0) # (3,)
        self.legal_moves = legal_move_lists[0]
        self.color = board.turn
        self.move_san = None
        self.p = None
        self.policy = None
        self.mcts = False

    def choose_move(self, policy, mcts=None):
        if policy.mcts and mcts is not None:
            self.get_mcts_pred(mcts)
        if len(self.legal_moves) == 0:
            self.move_san = 'Terminal Position'
            return None
        if policy.greedy:
            move_idx = torch.argmax(self.probs).item()
        else:
            move_idx = torch.multinomial(self.probs, 1).item()
        p = self.probs[move_idx].item()
        move = self.legal_moves[move_idx]
        self.move_san = self.board.san(move) # Convert to SAN format
        self.p = round(100 * p, 2)
        self.policy = policy
        return move 
    
    def get_mcts_pred(self, mcts):
        pis = mcts.run_mcts_search([self.board])
        legal_moves, pi = pis[0] 
        self.legal_moves = legal_moves
        self.probs = pi
        self.mcts = True
    
    def show_topk(self, k=3):
        if len(self.legal_moves) == 0:
            return "No legal moves available."
        if len(self.legal_moves) < k:
            k = len(self.legal_moves)
        probs, idx = torch.topk(self.probs, k=k)
        moves = [self.legal_moves[i] for i in idx]
        # show moves in standard notation
        san_moves = [self.board.san(move) for move in moves]
        probs = probs.tolist()
        probs = [round(100 * pi, 2) for pi in probs]
        return '\n'.join([
            f"{move}: {prob}%" for move, prob in zip(san_moves, probs)
        ])
    
    def show_eval(self):
        white_eval = round(100 * self.evals[2].item(), 2)
        black_eval = round(100 * self.evals[0].item(), 2)
        draw_eval = round(100 * self.evals[1].item(), 2)
        return f"({self.idx}) White: {white_eval}%, Draw: {draw_eval}%, Black: {black_eval}%"

@dataclass
class Policy:
    greedy: bool = True
    mcts: bool = False
    beam: bool = False

class GamePositions:
    def __init__(self, game_id, white_policy=None, black_policy=None):
        self.id = game_id
        self.white_policy = white_policy
        self.black_policy = black_policy
        self.positions = []
        self.result = None

    def append(self, position):
        self.positions.append(position)

class PlayGames:
    def __init__(self):
        self.model = None
        self.mcts = None
        self.games = {}

        self.white_policy = None
        self.black_policy = None

        self.setup_game()

    def setup_game(self, num_simulations=10):
        ##################################
        # GET CONFIGS AND CHECKPOINT
        ##################################
        with open("engine_config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        rl_config = {
            'mcts':{"num_simulations": num_simulations, "cpuct": 1.0, "temperature_start": 1.0,},
            'beam_search': {"depth": 5, "width": 5},
        }

        load_dotenv()
        CHECKPOINT_PATH = os.getenv("BASE_MODEL")

        ##################################
        # GET MODEL
        ##################################
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        config['train']['device'] = self.device
        self.model = ChessLightningModule.load_from_checkpoint(CHECKPOINT_PATH, config=config)
        self.model.eval() # Set model to evaluation mode
        self.model.to(self.device)
        
        ##################################
        # MCTS
        ##################################
        if self.white_policy.mcts or self.black_policy.mcts:
            self.mcts = BATCHED_MCTS(self.model, rl_config)

        if self.white_policy.beam or self.black_policy.beam:
            self.beam = BEAM(self.model, rl_config)

    def set_policy(self, 
                   white_greedy=True, 
                   black_greedy=True, 
                   white_mcts=False,
                   black_mcts=False,
                   white_beam=False,
                   black_beam=False):
        
        self.white_policy = Policy(greedy=white_greedy, mcts=white_mcts, beam=white_beam)
        self.black_policy = Policy(greedy=black_greedy, mcts=black_mcts, beam=black_beam)

    def play_game(self):    
        game_id = len(self.games) + 1
        game = GamePositions(game_id, self.white_policy, self.black_policy)
        # Start Postion, initialize game
        board = chess.Board()
        i = 0
        ACTIVE_GAME = True
        while ACTIVE_GAME:
            # Encode position and get predictions
            pred = batch_predict(self.model, [board], device=self.device) # (1, L), (1, 3), [List[chess.Move]]

            position = Position(board.copy(), pred, i)
            game.append(position)

            # Check if game is over
            if board.is_game_over():
                ACTIVE_GAME = False
                break
            
            #### WHITE MOVE ####
            if board.turn == chess.WHITE:
                move = position.choose_move(self.white_policy, mcts=self.mcts)

            #### BLACK MOVE ####
            else:
                move = position.choose_move(self.black_policy, mcts=self.mcts)

            board.push(move)  # Update the board with the chosen move
            i += 1

            if i >= 200: 
                print("Game Over: 200 moves reached.")
                break  # just in case
        
        game.result = board.result()
        self.games[game_id] = game
        print(f"Game Over: {board.result()}")
        return game

    def play_games(self, num_games = 1):
        for i in range(num_games):
            print(f"Playing Game {i + 1}/{num_games}...")
            self.play_game()
        print("All games played.")

    def play_parallel_games(self, num_games=10):
        active_games = []
        finished_games = []
        
        # Initialize games
        for game_id in range(1, num_games + 1):
            board = chess.Board()
            game = GamePositions(game_id, self.white_policy, self.black_policy)
            active_games.append((board, game, 0))  # board, game object, move counter

        i = 0
        while active_games:
            if i % 10 == 0:
                print(f"Move {i} of {len(active_games)} active games...")
            # Determine move policy based on ply (i)
            move_policy = self.white_policy if (i % 2 == 0) else self.black_policy

            boards = [b for (b, _, _) in active_games]
            preds = batch_predict(self.model, boards, device=self.device)  # (B, L), (B, 3), [List[chess.Move]]

            # If using MCTS for this move, run batch MCTS once
            mcts_pis = None
            if move_policy.mcts:
                mcts_pis = self.mcts.run_mcts_search(boards)

            next_active_games = []
            for idx, ((board, game, move_idx), pred) in enumerate(zip(active_games, zip(*preds))):
                position = Position(board.copy(), (pred[0].unsqueeze(0), pred[1].unsqueeze(0), [pred[2]]), move_idx)
                game.append(position)

                if board.is_game_over() or move_idx >= 200:
                    board_result = board.result() if board.is_game_over() else "*"
                    game.result = board_result
                    self.games[game.id] = game
                    print(f"Game {game.id} finished: {board_result}")
                    finished_games.append(game)
                    continue

                # If using MCTS, overwrite position probs with MCTS output
                if move_policy.mcts:
                    legal_moves, pi = mcts_pis[idx]
                    position.legal_moves = legal_moves
                    position.probs = pi
                    position.mcts = True

                move = position.choose_move(move_policy, mcts=None)

                if move is None:
                    game.result = "*"
                    self.games[game.id] = game
                    print(f"Game {game.id} finished unexpectedly.")
                    finished_games.append(game)
                else:
                    board.push(move)
                    next_active_games.append((board, game, move_idx + 1))

            active_games = next_active_games
            i += 1
        
        print("All games played.")
        return finished_games