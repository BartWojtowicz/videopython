import os
import requests
from pathlib import Path

import click

PEXELS_API_URL = "https://api.pexels.com/videos/search"


@click.command(help="Downloads videos with Pexels API.")
@click.option(
    "-s",
    "--search-query",
    help="Search query e.g. 'dogs playing in the snow'.",
    prompt="Search query",
    required=True,
    type=str,
)
@click.option(
    "-k",
    "--api-key",
    default=os.getenv("PEXELS_API_KEY"),
    help="Pexels API key.",
    required=True,
    type=str,
)
@click.option(
    "-d",
    "--download-dir",
    help="Directory to save downloaded videos.",
    prompt="Download directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True,
                    writable=True),
)
@click.option(
    "-n",
    "--n_videos",
    default=10,
    prompt="Number of videos",
    help="Number of videos to download. Defaults to 10.",
    type=int,
)
def download_videos(search_query: str, api_key: str, n_videos: int,
                    download_dir: str):
    params = {"query": search_query, "per_page": n_videos, "page": 1}
    headers = {"Authorization": api_key}

    response = requests.get(PEXELS_API_URL, headers=headers, params=params)
    data = response.json()

    # Download videos
    skipped = 0
    for video in data["videos"]:
        video_path = Path(download_dir) / f"{video['id']}.mp4"
        if video_path.exists():
            print(f"Video {video_path} already exists. Skipping...")
            skipped += 1
            continue
        video_url = video["video_files"][0]["link"]
        video_data = requests.get(video_url).content

        with open(video_path, "wb") as f:
            f.write(video_data)
        print(f"Video {video_path} downloaded.")


if __name__ == "__main__":
    download_videos()
