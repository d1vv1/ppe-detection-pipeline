import wandb
import os
import shutil
from ultralytics import YOLO

def train_model():
    """
    Trains the YOLOv8 model on the CSS dataset and logs to W&B.
    """
    print("Starting training process...")

    # Initialize Weights & Biases

    try:
        wandb.login()
        wandb.init(project="ppe-detection-pipeline", name="yolov8m-baseline-run")
        print("Weights & Biases initialized.")
    except Exception as e:
        print(f"Could not initialize W&B: {e}\nProceeding without logging.")

    # Define data path
    
    data_yaml_path = os.path.join(os.getcwd(), 'data', 'ppe-dataset', 'data.yaml')

    if not os.path.exists(data_yaml_path):
        print(f"Error: data.yaml not found at {data_yaml_path}")
        print("Please ensure your dataset is in 'data/ppe-dataset' and contains data.yaml")
        return

    # Load the pre-trained model
    # We use 'yolov8m.pt' for a good balance of speed and accuracy
    model = YOLO('yolov8m.pt')
    print("Loaded YOLOv8m pre-trained model.")

    # Train the model
    print("Starting model training...")
    results = model.train(
        data=data_yaml_path,
        epochs=25,
        imgsz=640,
        batch=8,
        name='yolov8m_ppe_detection'
    )

    print("Training complete.")
    
    # Save the best model to our '/models' directory
    try:
        # The best model is saved by Ultralytics in 'runs/detect/yolov8m_ppe_detection/weights/best.pt'
        source_model_path = os.path.join(os.getcwd(), 'runs', 'detect', 'yolov8m_ppe_detection', 'weights', 'best.pt')
        target_model_path = os.path.join(os.getcwd(), 'models', 'best.pt')

        # Ensure the /models directory exists
        os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)

        # Copy the file
        shutil.copyfile(source_model_path, target_model_path)
        print(f"Best model copied to {target_model_path}")

    except FileNotFoundError:
        print(f"Error: Could not find trained model at {source_model_path}")
    except Exception as e:
        print(f"Error copying model: {e}")

if __name__ == "__main__":
    train_model()