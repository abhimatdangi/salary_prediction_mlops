import os, shutil

def run(
    trained_model: str = "/opt/airflow/dags/artifacts/best_model.pkl",
    api_folder: str = "/opt/airflow/dags/api"
):
    """
    copy trained model to API folder.
    """
    os.makedirs(api_folder, exist_ok=True)
    deployed = os.path.join(api_folder, "salary_model.pkl")
    if not os.path.exists(trained_model):
        raise FileNotFoundError(f"Trained model not found at {trained_model}")
    shutil.copy(trained_model, deployed)
    print(f"Model copied to {deployed} for deployment.")
