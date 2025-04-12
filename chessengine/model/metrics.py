import torch

class MoveMetrics:

    @staticmethod
    def compute_all_metrics(logit_dict, lables, TRAINING=False):
        """
        Computes all defined metrics and returns a combined dictionary.
        """
        move_logits = logit_dict["move"]  # (B, L)
        eval_logits = logit_dict["eval"]  # (B, 3)
        true_index = lables["true_index"]  # (B,)
        eval = lables["eval"]  # (B, 1)

        prepared = MoveMetrics._prepare(move_logits, true_index)
        if prepared is None:
            return {}

        move_logits, true_index, probs, valid_mask, legal_counts = prepared
        metrics = {}

        metrics["move_accuracy"] = MoveMetrics.compute_move_accuracy(move_logits, true_index)
        metrics["true_prob"] = MoveMetrics.compute_true_move_prob(probs, true_index)

        if eval_logits is not None:
            metrics["eval_accuracy"] = MoveMetrics.eval_accuracy(eval_logits, eval)
            
        if not TRAINING:
            metrics.update(MoveMetrics.compute_topk_probs(probs, valid_mask, legal_counts))
            metrics.update(MoveMetrics.compute_topk_accuracy(move_logits, true_index, valid_mask, legal_counts))
        return metrics

    @staticmethod
    def _prepare(move_logits, true_index):
        """
        Args:
            move_logits (Tensor): logits for all legal moves (B, L)
            true_index (Tensor): index of ground truth move (B,)
            eval (Tensor): eval class (B, 1), 0 for black win, 1 for draw, 2 for white win
        Returns:
            move_logits (Tensor): logits for all legal moves (B', L)
            true_index (Tensor): index of ground truth move (B',)
            probs (Tensor): softmax probabilities for all legal moves (B', L)
            valid_mask (Tensor): mask padded (invalid) logits (B', L)
            legal_counts (Tensor): number of legal moves for each sample (B',)
        """
        # mask samples with no valid moves
        mask = (true_index != -1) # (B,)
        if not mask.any():
            return None
        move_logits = move_logits[mask] # (B', L)
        true_index = true_index[mask] # (B',)
        probs = torch.softmax(move_logits, dim=-1)
        valid_mask = ~torch.isinf(move_logits) # (B', L)
        legal_counts = valid_mask.sum(dim=-1) # (B',)
        return move_logits, true_index, probs, valid_mask, legal_counts

    @staticmethod
    def compute_move_accuracy(move_logits, true_index):
        pred_idx = move_logits.argmax(dim=-1)
        correct = (pred_idx == true_index)
        accuracy = 100 * correct.sum().float() / (true_index != -1).sum().float()
        return accuracy.item()

    @staticmethod
    def compute_true_move_prob(probs, true_index):
        """Computes the average softmax probability assigned to the correct move."""
        true_probs = probs.gather(1, true_index.unsqueeze(1))
        avg_prob = true_probs.mean().item()
        return avg_prob

    @staticmethod
    def compute_topk_probs(probs, valid_mask, legal_counts, ks=[1, 3, 5]):
        """
        Computes the top-k probability mass from the softmaxed move distribution.
        Also computes the average fraction of legal moves below 1% probability,
        ignoring padded (invalid) logits and skipping samples with < 20 legal moves.
        Returns:
            topk_dict (Dict[str, float]): Dictionary of top-k coverage percentages
        """
        LOW_PROB_THRESHOLD = 0.01
        topk_dict = {}
        for k in ks:
            topk_vals, _ = probs.topk(k, dim=-1)
            topk_sum = topk_vals.sum(dim=-1)
            avg_topk = topk_sum.mean().item()
            topk_dict[f"top{k}_prob"] = avg_topk

        valid_samples = legal_counts >= 20
        if valid_samples.any():
            selected_probs = probs[valid_samples]
            selected_valid_mask = valid_mask[valid_samples]
            selected_counts = legal_counts[valid_samples]
            low_mask = (selected_probs < LOW_PROB_THRESHOLD) & selected_valid_mask
            low_frac = low_mask.sum(dim=-1).float() / selected_counts
            avg_low_frac = low_frac.mean().item()
            topk_dict["lowprob_frac"] = avg_low_frac

        return topk_dict

    @staticmethod
    def compute_topk_accuracy(move_logits, true_index, valid_mask, legal_counts, ks=[3, 5]):
        """
        Computes the top-k accuracy, i.e., the percentage of times the correct move is among the top-k predicted.
        Includes all positions regardless of how many legal moves they have.
        Returns:
            topk_accs (Dict[str, float]): Dictionary of top-k accuracy values
        """
        topk_accs = {}
        for k in ks:
            topk_preds = move_logits.topk(k, dim=-1).indices  # (B, k)
            correct = (topk_preds == true_index.unsqueeze(1)).any(dim=-1)  # (B,)
            acc = 100 * correct.sum().float() / correct.numel()
            topk_accs[f"top{k}_acc"] = acc.item()
        return topk_accs
    
    @staticmethod
    def eval_accuracy(eval_logits, eval):
        """
        Computes the accuracy of the eval prediction.
        Args:
            eval_logits (Tensor): logits for eval (B, 1)
            eval (Tensor): ground truth eval (B, 1)
        Returns:
            accuracy (float): The accuracy of the eval prediction in percentage.
        """
        eval_pred = torch.argmax(eval_logits, dim=-1)
        eval_acc = (eval_pred == eval).float().mean()

        return eval_acc.item()


# def compute_move_accuracy(self, move_logits, true_index):
#     """
#     Computes the accuracy of the move prediction.
#     Args:
#         move_logits (Tensor): logits for all legal moves (B, L)
#         true_index (Tensor): index of ground truth move (B,)
#     Returns:
#         accuracy (float): The accuracy of the move prediction in percentage.
#     """
#     pred_idx = move_logits.argmax(dim=-1)  # (B,)
#     correct = (pred_idx == true_index)  # (B,)
#     accuracy = 100 * correct.sum().float() / (true_index != -1).sum().float() 
#     return accuracy.item()

# def compute_true_move_prob(self, move_logits, true_index):
#     """
#     Computes the average softmax probability assigned to the correct move.
#     Args:
#         move_logits (Tensor): logits for all legal moves (B, L)
#         true_index (Tensor): index of ground truth move (B,)
#     Returns:
#         avg_prob (float): Average predicted probability assigned to the correct move.
#     """
#     mask = (true_index != -1)                               # (B,)
#     move_logits = move_logits[mask]                         # (B', L)
#     true_index = true_index[mask]                           # (B',)

#     probs = torch.softmax(move_logits, dim=-1)              # (B', L)
#     true_probs = probs.gather(1, true_index.unsqueeze(1))   # (B', 1)
#     avg_prob = true_probs.mean().item()                     # Scalar
#     return avg_prob

# def compute_topk_probs(self, move_logits, true_index, ks=[1, 3, 5]):
#     """
#     Computes the top-k probability mass from the softmaxed move distribution.
#     Also computes the average fraction of legal moves below 1% probability,
#     ignoring padded (invalid) logits and skipping samples with < 20 legal moves.
#     Args:
#         move_logits (Tensor): logits for all legal moves (B, L)
#         true_index (Tensor): index of ground truth move (B,)
#         ks (List[int]): List of k-values to compute top-k probabilities
#     Returns:
#         topk_dict (Dict[str, float]): Dictionary of top-k coverage percentages
#     """
#     mask = (true_index != -1)
#     move_logits = move_logits[mask]  # (B', L)
#     probs = torch.softmax(move_logits, dim=-1)  # (B', L)
#     valid_mask = ~torch.isinf(move_logits)      # (B', L)
#     legal_counts = valid_mask.sum(dim=-1)       # (B',)

#     topk_dict = {}
#     for k in ks:
#         topk_vals, _ = probs.topk(k, dim=-1)  # (B', k)
#         topk_sum = topk_vals.sum(dim=-1)      # (B',)
#         avg_topk = topk_sum.mean().item()     # scalar
#         topk_dict[f"top{k}_prob"] = avg_topk

#     # Compute low-probability fraction for samples with at least 20 legal moves
#     valid_samples = legal_counts >= 20  # (B',)
#     if valid_samples.any():
#         selected_probs = probs[valid_samples]            # (B'', L)
#         selected_valid_mask = valid_mask[valid_samples]  # (B'', L)
#         selected_counts = legal_counts[valid_samples]    # (B'',)
#         low_mask = (selected_probs < 0.01) & selected_valid_mask
#         low_frac = low_mask.sum(dim=-1).float() / selected_counts  # (B'',)
#         avg_low_frac = low_frac.mean().item()
#         topk_dict["lowprob_frac"] = avg_low_frac

#     return topk_dict

# def compute_topk_accuracy(self, move_logits, true_index, ks=[3, 5]):
#     """
#     Computes the top-k accuracy, i.e., the percentage of times the correct move is among the top-k predicted.
#     Only includes positions with at least k valid moves.
#     Args:
#         move_logits (Tensor): logits for all legal moves (B, L)
#         true_index (Tensor): index of ground truth move (B,)
#         ks (List[int]): List of k values to compute top-k accuracy
#     Returns:
#         topk_accs (Dict[str, float]): Dictionary of top-k accuracy values
#     """
#     mask = (true_index != -1)
#     move_logits = move_logits[mask]  # (B', L)
#     true_index = true_index[mask]    # (B',)
#     valid_mask = ~torch.isinf(move_logits)  # (B', L)
#     legal_counts = valid_mask.sum(dim=-1)   # (B',)

#     topk_accs = {}
#     for k in ks:
#         # Select only samples with at least k legal moves
#         enough_moves = legal_counts >= k
#         if enough_moves.any():
#             logits_k = move_logits[enough_moves]
#             targets_k = true_index[enough_moves]
#             topk_preds = logits_k.topk(k, dim=-1).indices  # (B'', k)
#             correct = (topk_preds == targets_k.unsqueeze(1)).any(dim=-1)  # (B'',)
#             acc = 100 * correct.sum().float() / correct.numel()
#             topk_accs[f"top{k}_acc"] = acc.item()
#     return topk_accs
