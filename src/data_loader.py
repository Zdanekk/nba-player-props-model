from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import pandas as pd
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def find_player_id(player_name: str) -> int:
    matched_players = players.find_players_by_full_name(player_name)

    if not matched_players:
        raise ValueError(f"Player '{player_name}' not found.")

    return matched_players[0]["id"]


def load_player_games_by_id(player_id: int, season: str = "2023-24") -> pd.DataFrame:
    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season
    )

    df = gamelog.get_data_frames()[0]

    if df.empty:
        return pd.DataFrame()

    return df


def prepare_basic_player_dataset(player_name: str, season: str = "2023-24") -> pd.DataFrame:
    player_id = find_player_id(player_name)
    df = load_player_games_by_id(player_id, season=season)

    if df.empty:
        raise ValueError(f"No game data found for {player_name} in season {season}.")

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
    df["PLAYER_NAME"] = player_name
    df["PLAYER_ID"] = player_id
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    return df


def save_player_dataset(player_name: str, season: str = "2023-24", output_path: str = None) -> str:
    df = prepare_basic_player_dataset(player_name, season=season)

    safe_player_name = player_name.lower().replace(" ", "_")
    safe_season = season.replace("-", "_")

    if output_path is None:
        output_path = RAW_DATA_DIR / f"{safe_player_name}_{safe_season}_games.csv"
    else:
        output_path = Path(output_path)

    df.to_csv(output_path, index=False)
    return str(output_path)


def get_active_players() -> list[dict]:
    return players.get_active_players()


def download_multiple_players_data(
    season: str = "2023-24",
    max_players: int = 100,
    sleep_seconds: float = 0.7
) -> pd.DataFrame:
    active_players = get_active_players()[:max_players]

    all_dfs = []

    for i, player in enumerate(active_players, start=1):
        player_name = player["full_name"]
        player_id = player["id"]

        print(f"[{i}/{len(active_players)}] Downloading {player_name}...")

        try:
            df = load_player_games_by_id(player_id, season=season)

            if df.empty:
                print(f"  -> No data for {player_name}")
                continue

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
            df["PLAYER_NAME"] = player_name
            df["PLAYER_ID"] = player_id
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

            all_dfs.append(df)

        except Exception as e:
            print(f"  -> Error for {player_name}: {e}")

        time.sleep(sleep_seconds)

    if not all_dfs:
        raise ValueError("No player data downloaded.")

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)

    return final_df


def save_full_league_dataset(
    season: str = "2023-24",
    max_players: int = 100
) -> str:
    df = download_multiple_players_data(season=season, max_players=max_players)

    output_path = RAW_DATA_DIR / f"nba_players_{season.replace('-', '_')}_raw.csv"
    df.to_csv(output_path, index=False)

    return str(output_path)


if __name__ == "__main__":
    season = "2023-24"
    max_players = 400

    print(f"Downloading NBA data for {max_players} active players ({season})...")
    saved_path = save_full_league_dataset(season=season, max_players=max_players)
    print(f"Saved to: {saved_path}")