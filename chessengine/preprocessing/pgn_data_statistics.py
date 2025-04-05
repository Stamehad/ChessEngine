import argparse
import zstandard as zstd
import io
from collections import Counter
from tqdm import tqdm
import re

def extract_elos(game_lines):
    """Extract white and black ELOs from a list of PGN header lines."""
    white_elo, black_elo = None, None
    for line in game_lines:
        if line.startswith("[WhiteElo"):
            white_elo = int(re.findall(r"\d+", line)[0])
        elif line.startswith("[BlackElo"):
            black_elo = int(re.findall(r"\d+", line)[0])
    return white_elo, black_elo

def is_valid_game(game_lines, min_elo):
    """Returns True if both players have ELO ≥ min_elo."""
    white_elo, black_elo = extract_elos(game_lines)
    return white_elo is not None and black_elo is not None and min(white_elo, black_elo) >= min_elo

def main():
    parser = argparse.ArgumentParser(description="Collect ELO histogram from PGN .zst file")
    parser.add_argument("input_path", type=str, help="Path to .zst PGN file")
    parser.add_argument("--min_elo", type=int, default=2000, help="Minimum ELO to consider")
    parser.add_argument("--max_games", type=int, default=None, help="Optional maximum number of games to process")
    parser.add_argument("--output_path", type=str, default=None, help="Path to write filtered games")
    args = parser.parse_args()

    input_path = args.input_path
    min_elo = args.min_elo
    max_games = args.max_games

    #elo_counter = Counter()
    total_games = 0
    filtered_games = 0
    threshold_counts = {t: 0 for t in [2000, 2100, 2200, 2300, 2400, 2500]}
    save_filtered = args.output_path is not None

    try:
        with open(input_path, "rb") as compressed:
            if save_filtered:
                output_file = open(args.output_path, "w", encoding="utf-8")
            dctx = zstd.ZstdDecompressor()
            stream_reader = io.TextIOWrapper(dctx.stream_reader(compressed), encoding='utf-8', errors='ignore')
            lines = []
            progress = tqdm(unit="games", dynamic_ncols=True)

            for raw_line in stream_reader:
                if raw_line.startswith("[Event "):
                    if lines:
                        total_games += 1
                        if is_valid_game(lines, min_elo):
                            white_elo, black_elo = extract_elos(lines)
                            min_player_elo = min(white_elo, black_elo)

                            # for elo in (white_elo, black_elo):
                            #     binned_elo = 25 * (elo // 25)
                            #     elo_counter[binned_elo] += 1

                            for threshold in threshold_counts:
                                if min_player_elo >= threshold:
                                    threshold_counts[threshold] += 1

                            filtered_games += 1
                            if save_filtered:
                                output_file.write("".join(lines) + "\n\n")

                        progress.update(1)

                        if max_games is not None and total_games >= max_games:
                            break

                        lines = []

                lines.append(raw_line)

        # Handle last game if file doesn't end with new [Event
        if lines:
            total_games += 1
            if is_valid_game(lines, min_elo):
                white_elo, black_elo = extract_elos(lines)
                min_player_elo = min(white_elo, black_elo)

                # for elo in (white_elo, black_elo):
                #     binned_elo = 25 * (elo // 25)
                #     elo_counter[binned_elo] += 1

                for threshold in threshold_counts:
                    if min_player_elo >= threshold:
                        threshold_counts[threshold] += 1

                filtered_games += 1
                if save_filtered:
                    output_file.write("".join(lines) + "\n\n")

    except KeyboardInterrupt:
        print("\nAborted by user.")

    if save_filtered:
        output_file.close()

    print("\n========== SUMMARY ==========")
    print(f"Total games processed: {total_games:,}")
    print(f"Games passing filter (min ELO {min_elo}): {filtered_games:,}")
    print("\nGames with both players ≥ ELO thresholds:")
    for t in sorted(threshold_counts):
        print(f"  ≥ {t}: {threshold_counts[t]:,} games")

if __name__ == "__main__":
    main()
