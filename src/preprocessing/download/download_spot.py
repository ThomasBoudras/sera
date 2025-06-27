from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AI4Forest/Open-Canopy",
    repo_type="dataset",
    allow_patterns=["canopy_height/2021/spot/*", "canopy_height/2022/spot/*", "canopy_height/2023/spot/*"],  # Ne télécharge que ce dossier
    local_dir="/work/data/spot/2022"  # Dossier où stocker les fichiers
)
