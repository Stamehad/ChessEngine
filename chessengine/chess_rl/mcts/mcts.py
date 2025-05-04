from .search_tree import SearchForest
from chessengine.model.utils import Timer
import logging

logger = logging.getLogger(__name__)

class BATCHED_MCTS:
    """
    Monte Carlo Tree Search implementation using SearchForest and SearchTree.

    This class performs batched MCTS over a list of root boards, using a shared model
    and internal SearchTree objects for each position. It assumes that trees are independent
    and handles all model inference in batch. The search flow is:
      1. Initialize roots
      2. Run simulations in batch (select → evaluate → expand + backprop)
      3. Extract policy distributions from root visit counts
    """
    def __init__(self, model, config):
        self.model = model 
        self.config = config 
        self.device = next(model.parameters()).device
        
    def run_mcts_search(self, boards):
        num_simulations = self.config["mcts"]["num_simulations"]
        cpuct = self.config["mcts"]["cpuct"]
        temperature = self.config["mcts"]["temperature_start"]

        forest = SearchForest(boards, self.model) # search-trees for each board
        #with Timer("Init: evaluate + expand root"):
        forest.evaluate_leaves()
        forest.expand_and_backprop()

        for _ in range(num_simulations):
            #with Timer("Select leaves"):
            forest.select_leaves(cpuct)
            #with Timer("Evaluate leaves"):
            forest.evaluate_leaves()
            #with Timer("Expand and backprop"):
            forest.expand_and_backprop()

        #with Timer("Extract policies"):
        return forest.get_policies(temperature) # [(legal_moves, pi), ...]