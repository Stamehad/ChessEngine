from chessengine.chess_rl.beam_search.beam_tree import BeamTree

class BEAM:
    """
    Beam Search engine built on top of BeamTree.

    This class performs beam search starting from one or more root boards,
    using a shared model and internally managing BeamTree instances.
    The key steps are:
      1. Initialize root positions
      2. Perform beam search in batch mode (selection → evaluation → expansion → backpropagation)
      3. Extract best move sequences and evaluations

    The main outputs of run_beam_search() are:
        - best_moves_list: For each board, a sequence [m1, m2, ..., mk] representing the best branch found.
          Applying m1 to the root board produces a new board, then applying m2 to that board, and so on recursively.
          This sequence represents the principal variation (best sequence of moves).
        - evals: For each board, the evaluation at the leaf of the best sequence,
          corresponding to the expected outcome assuming optimal play along that branch.

    Optionally, coverage data (distributional statistics over move probabilities) can also be returned.
    """
    def __init__(self, model, config):
        self.model = model 
        self.config = config 
        self.device = next(model.parameters()).device
        self.k = config["width"] # number of branches per position
        self.depth = config["depth"] # search depth
        self.topk_schedule = config["topk_schedule"]

    def run_beam_search(self, boards, PAD_AND_CROP=True, COVERAGE_DATA=False): # [chess.Board]
        """
        Run beam search on the given board.
        """
        tree = BeamTree(self.model, device=self.device, topk_schedule=self.topk_schedule)
        tree.setup(boards)
        tree.expand_to_depth(depth=self.depth, k=self.k)
        tree.backpropagate()

        best_moves_list = tree.get_best_moves()
        if PAD_AND_CROP:
            best_moves_list = self._pad_and_crop(best_moves_list, crop=3) 
        evals = tree.roots_eval

        if COVERAGE_DATA:
            # Get coverage data
            coverage_data = tree.get_coverage_data()
            return best_moves_list, evals, coverage_data # [chess.Move]*B, [(3,)]*B, (B, topk)

        return best_moves_list, evals # [chess.Move]*B, [(3,)]*B
    
    def _pad_and_crop(self, moves_list, crop=None):
        """
        Pad the move list to ensure all lists have the same length.
        None moves will be used as identity moves.
        If crop is specified, truncate to that length after padding.
        """
        if not moves_list:
            return []

        max_len = max(len(moves) for moves in moves_list)
        if crop is not None:
            max_len = min(max_len, crop)

        padded_moves = [
            (moves + [None] * (max_len - len(moves)))[:max_len]
            for moves in moves_list
        ]

        return padded_moves