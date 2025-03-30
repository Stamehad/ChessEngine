import os
import re
import argparse
import json
import tqdm

def parse_pgn_file(filepath, min_time=300, max_games=None):
    """
    Fast PGN parser using regex to filter games by ELO without using python-chess.
    Returns a list of game dictionaries with 'pgn', 'white_elo', 'black_elo', 'result'.
    """
    
    games = []
    current_game = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith("[Event "):
                if current_game:
                    games.append("".join(current_game))
                    current_game = []
                    if max_games and len(games) >= max_games:
                        break
            current_game.append(line)
        if current_game:
            games.append("".join(current_game))

    print(f"Found {len(games)} games in the PGN file")
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
            time_control = headers.get('TimeControl', 0)
            time_control = int(time_control.split('+')[0]) if '+' in time_control else int(time_control)
            if time_control < min_time:
                continue

            result = headers.get('Result')
            if result not in ['1-0', '0-1', '1/2-1/2']:
                continue

            filtered.append({
                'pgn': '\n'.join(header_lines) + '\n\n' + '\n'.join(move_lines),
                'result': result
            })

            if max_games and len(filtered) >= max_games:
                break
        except Exception as e:
            # print("Skipped game due to parsing error:", e)
            # print("Headers:", headers)
            continue

    return filtered

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw PGN file")
    parser.add_argument("--output", type=str, required=True, help="Path to save filtered games (as .pt or .jsonl)")
    parser.add_argument("--min_time", type=int, default=300, help="Minimum time to include a game")
    parser.add_argument("--max_games", type=int, default=None, help="Optional limit on number of games")
    args = parser.parse_args()

    # print
    print(f"Filtering PGN file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Minimum time: {args.min_time}")
    print(f"Maximum games: {args.max_games}")


    filtered_games = parse_pgn_file(args.input, args.min_time, args.max_games)

    # Save as JSON lines
    with open(args.output, 'w', encoding='utf-8') as f:
        for game in filtered_games:
            f.write(json.dumps(game) + '\n')

    print(f"Saved {len(filtered_games)} filtered games to {args.output}")
