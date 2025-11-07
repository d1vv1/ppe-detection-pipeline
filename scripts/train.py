import wandb
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


def train_model(
    model_size='m',
    epochs=25,
    imgsz=640,
    batch=8,
    project_name="ppe-detection-pipeline",
    run_name=None,
    device=None,
    patience=50,
    save_period=10,
    val=True,
    plots=True,
    **kwargs
):
    """
    Trains the YOLOv8 model on the PPE dataset with comprehensive W&B logging.
    
    Args:
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        project_name: W&B project name
        run_name: W&B run name (auto-generated if None)
        device: Device to use ('cpu', 'cuda', '0', '1', etc.)
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
        val: Whether to run validation during training
        plots: Whether to save plots
        **kwargs: Additional training arguments
    """
    print("=" * 60)
    print("Starting PPE Detection Model Training")
    print("=" * 60)
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Define paths
    data_yaml_path = project_root / 'data' / 'ppe-dataset' / 'data.yaml'
    model_dir = project_root / 'models'
    model_dir.mkdir(exist_ok=True)
    
    # Validate data.yaml exists
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Error: data.yaml not found at {data_yaml_path}\n"
            "Please ensure your dataset is in 'data/ppe-dataset' and contains data.yaml"
        )
    
    # Initialize Weights & Biases with proper configuration
    wandb_initialized = False
    try:
        # Login to wandb (will use existing credentials if available)
        wandb.login()
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"yolov8{model_size}-ppe-{timestamp}"
        
        # Initialize wandb with comprehensive config
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "model_size": model_size,
                "model": f"yolov8{model_size}",
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "patience": patience,
                "save_period": save_period,
                "device": device if device else "auto",
                "data_yaml": str(data_yaml_path),
                **kwargs
            },
            # Track system metrics
            settings=wandb.Settings(
                _disable_stats=False,
                _disable_meta=False,
            )
        )
        wandb_initialized = True
        print(f"✓ Weights & Biases initialized: {project_name}/{run_name}")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize W&B: {e}")
        print("Proceeding without W&B logging. Metrics will still be tracked locally.")
    
    # Load the pre-trained model
    model_file = f'yolov8{model_size}.pt'
    print(f"Loading pre-trained model: {model_file}")
    model = YOLO(model_file)
    print(f"✓ Loaded YOLOv8{model_size} pre-trained model")
    
    # Prepare training arguments
    train_args = {
        "data": str(data_yaml_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "name": f'yolov8{model_size}_ppe_detection',
        "patience": patience,
        "save_period": save_period,
        "val": val,
        "plots": plots,
        "verbose": True,
        **kwargs
    }
    
    # Add device if specified
    if device:
        train_args["device"] = device
    
    # Enable wandb integration in Ultralytics (this ensures metrics are logged at every step/epoch)
    if wandb_initialized:
        train_args["project"] = project_name
        # Ultralytics will automatically log to wandb when project is set
    
    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")
    
    # Train the model
    print("Starting model training...")
    try:
        results = model.train(**train_args)
        print("\n✓ Training completed successfully!")
        
        # Log final metrics to wandb if initialized
        if wandb_initialized:
            # Ultralytics automatically logs metrics, but we can add custom ones
            if hasattr(results, 'results_dict'):
                wandb.log(results.results_dict)
            
            # Log model as artifact
            try:
                best_model_path = results.save_dir / "weights" / "best.pt"
                if best_model_path.exists():
                    artifact = wandb.Artifact(
                        f"yolov8{model_size}_ppe_best",
                        type="model",
                        description=f"Best YOLOv8{model_size} model for PPE detection"
                    )
                    artifact.add_file(str(best_model_path))
                    wandb.log_artifact(artifact)
                    print("✓ Model logged to W&B as artifact")
            except Exception as e:
                print(f"⚠ Warning: Could not log model artifact: {e}")
        
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        if wandb_initialized:
            wandb.log({"training_error": str(e)})
        raise
    
    # Save the best model to models directory
    try:
        # The best model is saved by Ultralytics in runs/detect/{name}/weights/best.pt
        run_name_dir = train_args["name"]
        source_model_path = project_root / 'runs' / 'detect' / run_name_dir / 'weights' / 'best.pt'
        target_model_path = model_dir / 'best.pt'
        
        if source_model_path.exists():
            shutil.copyfile(source_model_path, target_model_path)
            print(f"✓ Best model copied to {target_model_path}")
            
            # Also copy last.pt
            source_last_path = project_root / 'runs' / 'detect' / run_name_dir / 'weights' / 'last.pt'
            if source_last_path.exists():
                target_last_path = model_dir / 'last.pt'
                shutil.copyfile(source_last_path, target_last_path)
                print(f"✓ Last model copied to {target_last_path}")
        else:
            print(f"⚠ Warning: Could not find trained model at {source_model_path}")
            
    except Exception as e:
        print(f"⚠ Warning: Error copying model: {e}")
    
    # Finish wandb run
    if wandb_initialized:
        wandb.finish()
        print("✓ W&B run completed")
    
    print("\n" + "=" * 60)
    print("Training Pipeline Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # You can customize training parameters here
    train_model(
        model_size='m',  # Options: 'n', 's', 'm', 'l', 'x'
        epochs=25,
        imgsz=640,
        batch=8,
        device=None,  # Set to 'cpu', 'cuda', '0', etc. or None for auto
        patience=50,  # Early stopping patience
        save_period=10,  # Save checkpoint every N epochs
    )