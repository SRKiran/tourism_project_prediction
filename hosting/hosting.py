from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("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"))
api.upload_folder(
    folder_path="tourism_project_prediction/deployment",       # the local folder containing your files
    repo_id="SRKiran/tourism-package-prediction",   # the target HuggingFace Space
    repo_type="space",                              # dataset, model, or space
    path_in_repo="",                                # optional: subfolder path inside the repo
)
print("Deployment files uploaded to Hugging Face Space successfully.")
