# test_stopping_conditions.py
import sys
import torch
import chess
from pytorchchess import TorchBoard
import pytorchchess.utils.constants as constants

class TestStoppingConditions:
    
    def test_checkmate_detection(self):
        """Test checkmate positions"""
        # Fool's mate
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 1 2")
        board.push_san("Qh4#")  # Checkmate
        
        torch_board = TorchBoard.from_board_list([board], device='cpu')
        terminal, results = torch_board.is_game_over()
        
        assert terminal[0] == True, "Should detect checkmate"
        assert results[0] == -1, "Black should win (white is checkmated)"
    
    def test_stalemate_detection(self):
        """Test stalemate positions"""
        # Classic stalemate position
        board = chess.Board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")  # Black is stalemated
        
        torch_board = TorchBoard.from_board_list([board], device='cpu')
        terminal, results = torch_board.is_game_over()
        
        assert terminal[0] == True, "Should detect stalemate"
        assert results[0] == 0, "Should be a draw"
    
    def test_fifty_move_rule(self):
        """Test fifty-move rule"""
        board = chess.Board()
        
        # Create position with high fifty-move counter
        torch_board = TorchBoard.from_board_list([board], device='cpu')
        torch_board.state.fifty_move_clock[0] = 100  # Exactly at limit
        
        terminal, results = torch_board.is_game_over(enable_fifty_move_rule=True)
        
        assert terminal[0] == True, "Should detect fifty-move rule"
        assert results[0] == 0, "Should be a draw"
    
    def test_ply_limit(self):
        """Test maximum ply limit"""
        board = chess.Board()
        torch_board = TorchBoard.from_board_list([board], device='cpu')
        torch_board.state.plys[0] = 150  # Set high ply count
        
        terminal, results = torch_board.is_game_over(max_plys=100)
        
        assert terminal[0] == True, "Should detect ply limit exceeded"
        assert results[0] == 0, "Should be a draw"
    
    def test_threefold_repetition(self):
        """Test threefold repetition detection"""
        # Start with a position that can easily repeat
        board = chess.Board()
        torch_board = TorchBoard.from_board_list([board], device='cpu')
        # Simulate the same position occurring multiple times
        initial_hash = torch_board._compute_position_hash()[0]
        
        # Manually add to history to simulate repetitions
        torch_board._initialize_position_history()
        fake_hashes = torch.tensor([initial_hash, initial_hash, initial_hash], device=torch_board.device)
        
        for i in range(3):
            counts = torch_board._add_positions_to_history(fake_hashes[[i]])
        
        terminal, results = torch_board.is_game_over(enable_threefold_repetition=True)
        assert results[0] == 0, "Threefold repetition should be a draw"
    
def test_memory_usage():
    """Test memory usage with large batches"""
    #constants.move_constants_to('mps')  # Ensure constants are on the correct device
    # Create large batch
    board = chess.Board()
    large_batch = [board] * 1000
    torch_board = TorchBoard.from_board_list(large_batch, device='cpu')
    
    # Test memory overview
    print("Memory usage for B=1000:")
    torch_board.print_memory_overview(normalize_to_b=1000)
    
    # Test threefold repetition memory
    if hasattr(torch_board, 'position_history'):
        initial_memory = torch_board.position_history.numel() * torch_board.position_history.element_size()
        
        # Add some positions
        for _ in range(10):
            hash_vals = torch_board._compute_position_hash()
            torch_board._add_positions_to_history(hash_vals)
        
        final_memory = torch_board.position_history.numel() * torch_board.position_history.element_size()
        print(f"Memory growth: {final_memory - initial_memory} bytes")

def test_threefold_repetition_performance():
    """Test threefold repetition doesn't slow down significantly"""
    import time
    
    board = chess.Board()
    torch_board = TorchBoard.from_board_list([board] * 1000, device='cpu')
    
    # Test without threefold repetition
    start = time.time()
    for _ in range(100):
        terminal, results = torch_board.is_game_over(enable_threefold_repetition=False)
    baseline_time = time.time() - start
    
    # Test with threefold repetition
    start = time.time()
    for _ in range(100):
        terminal, results = torch_board.is_game_over(enable_threefold_repetition=True)
    repetition_time = time.time() - start
    
    overhead = (repetition_time - baseline_time) / baseline_time
    print(f"Threefold repetition overhead: {overhead:.1%}")
    
    # Should be reasonable overhead (adjust threshold as needed)
    assert overhead < 0.5, f"Threefold repetition overhead too high: {overhead:.1%}"
    
def test_famous_games():
    """Test stopping conditions on famous game endings"""
    
    # Test immortal game ending (checkmate)
    
    # Final position of Anderssen vs Kieseritzky
    board = chess.Board("r1bk3r/p2pBpNp/n4n2/1p1NP2P/6P1/3P4/P1P1K3/q5b1 b - - 1 23")
    torch_board = TorchBoard.from_board_list([board], device='cpu')
    
    terminal, results = torch_board.is_game_over()
    assert terminal[0] == True, "Should detect checkmate in immortal game"
    
    # Test endgame with insufficient material

    # King vs King
    board = chess.Board("8/8/8/8/8/8/3k4/3K4 w - - 0 1")
    torch_board = TorchBoard.from_board_list([board], device='cpu')
    
    terminal, results = torch_board.is_game_over(enable_insufficient_material=True)
    assert terminal[0] == True, "K vs K should be insufficient material"
    assert results[0] == 0, "Should be a draw"
    
def test_batch_conditions():
    """Test stopping conditions work correctly with batched positions"""
    
    # Create a batch with different ending types
    boards = [
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 1 2"),  # Normal position
        chess.Board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1"),  # Stalemate
        chess.Board("8/8/8/8/8/8/3k4/5K2 w - - 0 1"),  # Insufficient material
    ]
    
    # Set up different conditions
    torch_board = TorchBoard.from_board_list(boards, device='cpu')
    torch_board.state.fifty_move_clock[1] = 100  # Fifty-move rule for position 1
    terminal, results = torch_board.is_game_over(
        enable_fifty_move_rule=True,
        enable_insufficient_material=True
    )
    
    # Check that only terminal positions are detected
    expected_terminal = torch.tensor([False, True, True])
    assert torch.equal(terminal, expected_terminal), f"Expected {expected_terminal}, got {terminal}"
    
    # Check results are draws for the terminal positions
    assert all(r == 0 for r in results), "All terminal conditions should be draws"

def test_edge_cases():
    """Test edge cases and error conditions"""
    
    """Test what happens with hash collisions"""
    board = chess.Board()
    torch_board = TorchBoard.from_board_list([board], device='cpu')
    
    # Force a hash collision by manipulating position history
    torch_board._initialize_position_history()
    
    # Add same hash multiple times manually
    same_hash = torch.tensor([12345], device=torch_board.device, dtype=torch.long)
    for _ in range(4):  # More than 3 times
        counts = torch_board._add_positions_to_history(same_hash)
    
    assert counts[0] >= 3, "Should detect repetition even with manual collision"
    
    
    """Test that disabled conditions don't interfere"""
    # Position that would trigger multiple conditions
    board = chess.Board("k7/2Q5/1K6/8/8/8/8/8 b - - 99 1")  # Stalemate + near fifty-move
    torch_board = TorchBoard.from_board_list([board], device='cpu')
    
    # Test with all conditions disabled except standard
    terminal, results = torch_board.is_game_over(
        enable_fifty_move_rule=False,
        enable_insufficient_material=False,
        enable_threefold_repetition=False
    )
    
    assert terminal[0] == True, "Should still detect stalemate"
    assert results[0] == 0, "Should be a draw"


# run_stopping_condition_tests.py
def run_all_tests():
    """Run comprehensive test suite"""
    
    test_cases = [
        TestStoppingConditions().test_checkmate_detection,
        TestStoppingConditions().test_stalemate_detection,
        TestStoppingConditions().test_fifty_move_rule,
        TestStoppingConditions().test_ply_limit,
        TestStoppingConditions().test_threefold_repetition,
        test_famous_games,
        test_batch_conditions,
        test_memory_usage,
        test_threefold_repetition_performance,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_cases:
        try:
            print(f"Running {test_func.__name__}...")
            test_func()
            print(f"✅ {test_func.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)