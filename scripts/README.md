#  How to use the scripts in this folder

## 1. [Script for downloading videos with Pexels API](./download_videos.py)
First, you need to set up our Pexels account and get the API key from https://www.pexels.com/api/.

Basic usage:
```bash
python scripts/download_videos.py \
    -s "dogs playing in the snow" \
    -k <PEXELS_API_KEY> \
    -d data/downloaded \
    -n 10
```
