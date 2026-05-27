from roboflow import Roboflow
import os
import yaml

API_KEY = "ORwLOc6tOmQ0vYPEpsKs"  # <--- PUT YOUR KEY HERE
# DOWNLOAD_PATH = "./road_hazard_dataset" # <--- REMOVED

try:
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("cse400b").project("pothole-and-speed-breaker-dataset-400b")
    
    print("Downloading dataset 'Pothole and Speed Breaker Dataset-400B'...")
    
    # <--- MODIFIED
    # We remove the 'location' parameter.
    # The library will now create a folder in your current directory
    # (D:\pothole_detection\) named after the project.
    dataset = project.version(2).download("yolov8")
    
    print(f"\nDataset downloaded successfully.")
    print(f"The data is in this folder: {dataset.location}")

    # Check the data.yaml file in the *actual* download location
    yaml_path = os.path.join(dataset.location, "data.yaml")
    
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            if 'names' in data_config:
                print(f"Classes in this dataset: {data_config['names']}")
    else:
        print(f"Warning: data.yaml not found at {yaml_path}.")

    print("\nThis dataset is ready for training.")
    print(f"IMPORTANT: Now, open 'train.py' and change 'dataset_dir' to: '{dataset.location}'")


except Exception as e:
    print(f"\nAn error occurred:")
    print(e)