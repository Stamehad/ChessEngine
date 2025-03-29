import os
import re
import argparse
import json
import tqdm

def parse_pgn_file(filepath, min_elo=2200, max_games=None):
    """
    Fast PGN parser using regex to filter games by ELO without using python-chess.
    Returns a list of game dictionaries with 'pgn', 'white_elo', 'black_elo', 'result'.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Read and group games using a line-by-line parser that collects full blocks
    lines = raw.strip().splitlines()
    games = []
    current_game = []

    for line in lines:
        if line.startswith("[Event "):
            if current_game:
                games.append("\n".join(current_game))
                current_game = []
        current_game.append(line)
    if current_game:
        games.append("\n".join(current_game))

    filtered = []

    for game in tqdm.tqdm(games, desc="Filtering games"):
        header_lines = []
        move_lines = []
        for line in game.strip().splitlines():
            if line.startswith("["):
                header_lines.append(line)
            elif line.strip():  # skip empty lines
                move_lines.append(line)
        headers = dict(re.findall(r'\[(\w+)\s+"(.*?)"\]', "\n".join(header_lines)))
        try:
            white_elo = int(headers.get('WhiteElo', 0))
            black_elo = int(headers.get('BlackElo', 0))
            if min(white_elo, black_elo) < min_elo:
                continue

            result = headers.get('Result')
            if result not in ['1-0', '0-1', '1/2-1/2']:
                continue

            filtered.append({
                'pgn': f"{'\n'.join(header_lines)}\n\n{'\n'.join(move_lines)}",
                'white_elo': white_elo,
                'black_elo': black_elo,
                'result': result
            })

            if max_games and len(filtered) >= max_games:
                break
        except Exception:
            continue

    return filtered

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw PGN file")
    parser.add_argument("--output", type=str, required=True, help="Path to save filtered games (as .pt or .jsonl)")
    parser.add_argument("--min_elo", type=int, default=2000, help="Minimum ELO to include a game")
    parser.add_argument("--max_games", type=int, default=None, help="Optional limit on number of games")
    args = parser.parse_args()

    filtered_games = parse_pgn_file(args.input, args.min_elo, args.max_games)

    # Save as JSON lines
    with open(args.output, 'w', encoding='utf-8') as f:
        for game in filtered_games:
            f.write(json.dumps(game) + '\n')

    print(f"Saved {len(filtered_games)} filtered games to {args.output}")
