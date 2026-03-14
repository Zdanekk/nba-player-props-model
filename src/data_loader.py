from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import pandas as pd
import time


def find_player_id(player_name: str) -> int:
    """
    Find NBA player ID by full player name.
    Example: 'LeBron James'
    """
    matched_players = players.find_players_by_full_name(player_name)

    if not matched_players:
        raise ValueError(f"Player '{player_name}' not found.")

    return matched_players[0]["id"]


def load_player_games(player_name: str, season: str = "2023-24") -> pd.DataFrame:
    """
    Load game logs for a given player and season.
    """
    player_id = find_player_id(player_name)

    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season
    )

    df = gamelog.get_data_frames()[0]

    if df.empty:
        raise ValueError(f"No game data found for {player_name} in season {season}.")

    return df


def prepare_basic_player_dataset(player_name: str, season: str = "2023-24") -> pd.DataFrame:
    """
    Load and prepare a basic player dataset with selected columns.
    """
    df = load_player_games(player_name, season=season)

    selected_columns = [
        "Game_ID",
        "GAME_DATE",
        "MATCHUP",
        "WL",
        "MIN",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "REB",
        "AST",
        "TOV",
        "STL",
        "BLK",
        "PTS",
        "PLUS_MINUS"
    ]

    df = df[selected_columns].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    return df


def save_player_dataset(player_name: str, season: str = "2023-24", output_path: str = None) -> str:
    """
    Save prepared player dataset to CSV.
    """
    df = prepare_basic_player_dataset(player_name, season=season)

    safe_player_name = player_name.lower().replace(" ", "_")
    safe_season = season.replace("-", "_")

    if output_path is None:
        output_path = f"data/raw/{safe_player_name}_{safe_season}_games.csv"

    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    player = "LeBron James"
    season = "2023-24"

    print(f"Downloading data for {player} ({season})...")
    saved_path = save_player_dataset(player, season)
    print(f"Saved to: {saved_path}")