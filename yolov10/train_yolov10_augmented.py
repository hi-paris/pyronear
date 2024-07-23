# train.py
"""
Goal: Yolov10 model trained on Pyronear Dataset with parameters of configuration
saved thanks to mlflow. This code outputs the artifacts of the top model in aws.
"""

# 01 - Import Libraries

import argparse
import os
import shutil
import subprocess
import random
import numpy as np

import boto3
import mlflow
import pandas as pd
import yaml

from IPython.display import Image, display
from ultralytics.models import YOLO

# 02 Define methods used later on

def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")

def get_git_revision_short_hash():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")

def find_and_concatenate_csvs(root_dir, output_file):
    all_dfs = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "results.csv":
                path = os.path.join(root, file)
                print(path)
                df = pd.read_csv(path)
                df["Path"] = path
                all_dfs.append(df)

    concatenated_df = pd.concat(all_dfs, ignore_index=True)
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved as {output_file}")
    return concatenated_df

def upload_directory_to_s3(bucket_name, directory_name, s3_folder):
    s3_client = boto3.client(
        "s3", aws_access_key_id="your_access_key_id", aws_secret_access_key="your_secret_access_key"
    )
    for subdir, dirs, files in os.walk(directory_name):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, "rb") as data:
                s3_client.upload_fileobj(data, bucket_name, s3_folder + "/" + full_path[len(directory_name) + 1 :])
    return print("mlruns uploaded on aws")

# Define the random search space and functions
space = {
    "model_type": np.array(["yolov10n"]),
    "epochs": np.linspace(10, 30, 1, dtype=int),
    "patience": np.linspace(10, 50, 10, dtype=int),
    "imgsz": np.array([320, 640, 1024], dtype=int),
    "batch": np.array([16, 32, 64]),
    "optimizer": np.array(
        [
            "SGD",
            "Adam",
            "AdamW",
            "NAdam",
            "RAdam",
            "RMSProp",
            "auto",
        ]
    ),
    "lr0": np.logspace(
        np.log10(0.0001),
        np.log10(0.03),
        base=10,
        num=50,
    ),
    "lrf": np.logspace(
        np.log10(0.001),
        np.log10(0.01),
        base=10,
        num=50,
    ),
    "mixup": np.array([0, 0.2]),
    "close_mosaic": np.linspace(0, 35, 10, dtype=int),
    "degrees": np.linspace(0, 10, 10),
    "translate": np.linspace(0, 0.4, 10),
    "scale": np.linspace(0.5, 1.5, 10),
    "shear": np.linspace(0, 10, 10),
    "fliplr": np.array([0, 0.5]),
    "flipud": np.array([0, 0.5]),
    "mosaic": np.array([0, 1.0])
}

def draw_configuration(space, random_seed=0):
    random.seed(random_seed)
    return {k: random.choice(v).item() for k, v in space.items()}

def draw_n_random_configurations(space, n, random_seed=0):
    random.seed(random_seed)
    return [draw_configuration(space, random_seed=random.random()) for _ in range(n)]

# 03 Input Configuration

parser = argparse.ArgumentParser(description="Load YAML configuration files for YOLO model and data")
parser.add_argument("--data_config", type=str, required=True, help="Path to the data configuration YAML file")
parser.add_argument("--model_config", type=str, required=True, help="Path to the model configuration YAML file")
args = parser.parse_args()

# Generate a random configuration
random_config = draw_configuration(space, random_seed=42)

# Assign random configuration parameters to yolo_params
yolo_params = {
    "model_type": random_config["model_type"],
    "epochs": random_config["epochs"],
    "imgsz": random_config["imgsz"],
    "batch": random_config["batch"],
    "optimizer": random_config["optimizer"],
    "learning_rate": random_config["lr0"],
    "lrf": random_config["lrf"],
    "mixup": random_config["mixup"],
    "close_mosaic": random_config["close_mosaic"],
    #Data augmentation
    "degrees": random_config["degrees"],
    "translate": random_config["translate"],
    "scale": random_config["scale"],
    "shear": random_config["shear"],
    "fliplr": random_config["fliplr"],
    "flipud": random_config["flipud"],
    "mosaic": random_config["mosaic"],
    "experiment_name": "random_search_experiment",
    "pretrained": True,
}

# Load the data configuration yaml file
with open(args.data_config) as f:
    data_params = yaml.safe_load(f)
print("Data Parameters:", data_params)

# Load the model configuration yaml file
with open(args.model_config) as f:
    yolo_params_from_file = yaml.safe_load(f)
print("Model Parameters from file:", yolo_params_from_file)

# Overwrite some parameters with those from the random configuration
yolo_params.update(yolo_params_from_file)

# 04 Training the YOLO Model

print("YOLOv10 PARAMETERS:")
print(f"""model: {yolo_params['model_type']}""")
print(f"imgsz: {yolo_params['imgsz']}""")
print(f"lr0: {yolo_params['learning_rate']}""")
print(f"batch: {yolo_params['batch']}")
print(f"name: {yolo_params['experiment_name']}")
print(yolo_params)

model = YOLO(yolo_params["model_type"])

EXPERIMENT_NAME = "pyronear-v10"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)

dirpath = os.path.join("./runs/detect/", yolo_params["experiment_name"])
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="pyronear_yolov10") as dl_model_tracking_run:
    tags = {"Model": "Yolov10", "User": "maevanguessan", "Approach": "best_model"}
    mlflow.set_tags(tags)

    model.train(
        data=args.data_config,
        imgsz=yolo_params["imgsz"],
        batch=yolo_params["batch"],
        epochs=yolo_params["epochs"],
        optimizer=yolo_params["optimizer"],
        lr0=yolo_params["learning_rate"],
        pretrained=yolo_params["pretrained"],
        name=yolo_params["experiment_name"],
        seed=0,
        mixup=yolo_params["mixup"],
        close_mosaic=yolo_params["close_mosaic"],
        degrees=yolo_params["degrees"],
        translate=yolo_params["translate"],
        scale=yolo_params["scale"],
        shear=yolo_params["shear"],
        fliplr=yolo_params["fliplr"],
        flipud=yolo_params["flipud"],
        mosaic=yolo_params["mosaic"]
    )
    
    model.val()

    commit_hash = get_git_revision_hash()
    mlflow.log_param("git_commit_hash", commit_hash)

    installed_packages = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
    mlflow.log_param("dependencies", installed_packages)

path = f"./runs/detect/{yolo_params['experiment_name']}"
confusion_matrix = Image(os.path.join(path, "confusion_matrix_normalized.png"), width=800, height=600)
display(confusion_matrix)

val_images = Image(os.path.join(path, "val_batch0_labels.jpg"), width=800, height=600)
display(val_images)

run_id = dl_model_tracking_run.info.run_id
print("run_id: {}; lifecycle_stage: {}".format(run_id, mlflow.get_run(run_id).info.lifecycle_stage))

logged_model = f"runs:/{run_id}/model"
model_registry_version = mlflow.register_model(logged_model, "pyronear_dl_model")
print(f"Model Name: {model_registry_version.name}")
print(f"Model Version: {model_registry_version.version}")

output_csv_file = "concatenated_results.csv"
df = find_and_concatenate_csvs("../../mlartifacts", output_csv_file)
df.columns = df.columns.str.replace(" ", "")
df = df.reset_index(drop=True)
sorted_df = df.sort_values("metrics/mAP50-95(B)", ascending=False)
print(sorted_df.head())
print("Top metrics/mAP50-95(B):  ")
print("    ")
print("    ")
print(list(sorted_df["metrics/mAP50-95(B)"][0:1])[0])
print("    ")
print("    ")
print("Obtained with artefact: ", list(sorted_df["Path"][0:1])[0])

final_model = list(sorted_df["Path"][0:1])[0]

bucket_name = "pyronear-v10"
local_directory = "../../mlruns"
s3_folder = "output/mlruns"

upload_directory_to_s3(bucket_name, local_directory, s3_folder)
