"""Upload files ke HuggingFace Space."""
import os, sys
from huggingface_hub import HfApi

token   = os.environ.get("HF_TOKEN")
repo_id = "ggapar/DFK1-Text-Classify"

if not token:
    print("ERROR: HF_TOKEN tidak ditemukan.")
    sys.exit(1)

api = HfApi(token=token)

for filename in ["gradio_app.py", "requirements.txt", "README.md"]:
    if not os.path.exists(filename):
        print(f"SKIP: {filename} tidak ada")
        continue
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"sync: {filename} from GitHub Actions",
    )
    print(f"OK: {filename} uploaded")

print("Sync selesai.")
