from huggingface_hub import snapshot_download
import os

def download_dataset(repo_id, local_dir, revision='main', hf_token=None):
    print(f"start downloading from '{repo_id}' to '{local_dir}'...")
    try:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            revision=revision,
            resume_download=True
        )
        print(f"Successfully download to: {os.path.abspath(downloaded_path)}")
    except Exception as e:
        print(f"\nDownload failed: {e}")


repo_id_list = [
    'sapphirex/RAVine-nuggets',
    'sapphirex/RAVine-dense-index',
    'sapphirex/RAVine-qrels',
    'sapphirex/RAVine-logs',
    'sapphirex/RAVine-mapper',
]

local_dir = './data'

for repo_id in repo_id_list:
    download_dataset(repo_id, local_dir)
