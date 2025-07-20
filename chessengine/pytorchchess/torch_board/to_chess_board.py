import chess # type: ignore
import chess.svg # type: ignore
from IPython.display import SVG, display
from pytorchchess.utils.board_from_fen import tensor_to_fen
from pytorchchess.utils import to_chess_move

class ToChessBoard:
    
    def to_chess_board(self, index=0, flip=True):
        """
        Convert the TorchBoard instance at a given index to a python-chess Board.
        """
        piece_tensor = self.board_tensor[index].cpu()  # (8, 8)
        side = self.state.side_to_move[index].item()
        castling = self.state.castling[index].tolist()
        ep = self.state.ep[index].item()
        last_move = self.state.previous_moves[index] if self.state.previous_moves is not None else None

        if last_move is not None:
            last_move = to_chess_move(last_move)

        fen = tensor_to_fen(piece_tensor, side, castling, ep)
        return chess.Board(fen), last_move

    def render(self, index=0, flip=False):
        """
        Print a visual board representation using python-chess.
        Uses SVG rendering if in a Jupyter environment; falls back to ASCII.
        """
        assert 0 <= index < len(self.board_tensor), "Index out of bounds for board_tensor"
        board, lastmove = self.to_chess_board(index, flip=flip)
        try:
            orientation = chess.WHITE
            svg = chess.svg.board(board, lastmove=lastmove, orientation=orientation, size=400)
            display(SVG(svg))
        except ImportError:
            print(board)

