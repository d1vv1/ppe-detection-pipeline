import wandb
import os
import shutil
import time
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
    save_period=1,  # Changed to 1 for Colab safety - saves every epoch
    val=True,
    plots=True,
    resume=False,  # Resume from last checkpoint
    google_drive_backup=None,  # Path to Google Drive for backup (e.g., '/content/drive/MyDrive/ppe-models')
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
        save_period: Save checkpoint every N epochs (default: 1 for Colab safety)
        val: Whether to run validation during training (default: True)
        plots: Whether to save plots
        resume: Resume training from last checkpoint (default: False)
        google_drive_backup: Path to Google Drive folder for checkpoint backup
        **kwargs: Additional training arguments
    """
    print("=" * 60)
    print("Starting PPE Detection Model Training")
    print("=" * 60)
    
    # Get project root directory - handle both script execution and Colab
    if '__file__' in globals():
        project_root = Path(__file__).parent.parent
    else:
        # In Colab or when __file__ is not available
        project_root = Path(os.getcwd())
        # Try to detect if we're in the project root
        if not (project_root / 'data' / 'ppe-dataset' / 'data.yaml').exists():
            # Maybe we're already in the project root
            pass
    
    os.chdir(project_root)
    print(f"Working directory: {project_root}")
    
    # Define paths
    data_yaml_path = project_root / 'data' / 'ppe-dataset' / 'data.yaml'
    model_dir = project_root / 'models'
    model_dir.mkdir(exist_ok=True)
    
    # Validate data.yaml exists
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Error: data.yaml not found at {data_yaml_path}\n"
            "Please ensure your dataset is in 'data/ppe-dataset' and contains data.yaml\n"
            f"Current working directory: {os.getcwd()}"
        )
    
    # Verify validation dataset exists
    val_images_path = project_root / 'data' / 'ppe-dataset' / 'valid' / 'images'
    if val_images_path.exists():
        val_count = len(list(val_images_path.glob('*')))
        print(f"✓ Validation dataset found: {val_count} files in {val_images_path}")
    else:
        print(f"⚠ Warning: Validation directory not found at {val_images_path}")
        if val:
            print("  Validation is enabled but dataset may not be found. Training may fail.")
    
    # Set up Google Drive backup if provided
    drive_backup_dir = None
    if google_drive_backup:
        drive_backup_dir = Path(google_drive_backup)
        drive_backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Google Drive backup enabled: {drive_backup_dir}")
    
    # Initialize Weights & Biases with proper configuration
    wandb_initialized = False
    api_key = os.environ.get('WANDB_API_KEY')  # Get API key first for use in exception handler
    
    try:
        # Check for API key in environment (for Colab)
        if api_key:
            wandb.login(key=api_key)
            print("✓ W&B API key found in environment")
        else:
            # Try to login with existing credentials
            try:
                wandb.login()
            except Exception as e:
                print(f"⚠ W&B login failed: {e}")
                print("  Set WANDB_API_KEY environment variable for Colab")
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"yolov8{model_size}-ppe-{timestamp}"
        
        # Initialize wandb with config
        # CRITICAL: Initialize wandb BEFORE model.train() so Ultralytics can detect it
        # Use minimal configuration for maximum compatibility
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
                "val": val,
                "resume": resume,
                **kwargs
            }
        )
        
        wandb_initialized = True
        print(f"✓ Weights & Biases initialized: {project_name}/{run_name}")
        if wandb.run:
            print(f"  View run at: {wandb.run.url}")
    except Exception as e:
        error_msg = str(e)
        print(f"⚠ Warning: Could not initialize W&B manually: {error_msg}")
        
        # Check if it's an authentication error
        if "401" in error_msg or "not logged in" in error_msg.lower() or "PERMISSION_ERROR" in error_msg:
            print("  Authentication issue detected. Trying to re-login...")
            try:
                # Try to force re-login with relogin=True
                if api_key:
                    wandb.login(key=api_key, relogin=True)
                    # Wait a moment for login to complete
                    time.sleep(1)
                    # Try init again after re-login
                    if run_name is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        run_name = f"yolov8{model_size}-ppe-{timestamp}"
                    wandb.init(project=project_name, name=run_name)
                    wandb_initialized = True
                    print("✓ W&B initialized after re-login")
                else:
                    print("  No API key available for re-login. Ultralytics will attempt auto-initialization.")
            except Exception as retry_error:
                print(f"  Re-login failed: {retry_error}")
                print("  Ultralytics will attempt to initialize wandb automatically during training.")
        else:
            print("  Ultralytics will attempt to initialize wandb automatically during training.")
            print("  Metrics will be logged to W&B if Ultralytics successfully initializes it.")
    
    # Check for resume checkpoint
    run_name_dir = f'yolov8{model_size}_ppe_detection'
    last_checkpoint_path = project_root / 'runs' / 'detect' / run_name_dir / 'weights' / 'last.pt'
    
    # Load model - either resume from checkpoint or start fresh
    if resume:
        # Try to resume from local checkpoint
        if last_checkpoint_path.exists():
            print(f"✓ Resuming training from checkpoint: {last_checkpoint_path}")
            model = YOLO(str(last_checkpoint_path))
        elif drive_backup_dir and (drive_backup_dir / 'last.pt').exists():
            # Try to resume from Google Drive
            drive_checkpoint = drive_backup_dir / 'last.pt'
            print(f"✓ Resuming from Google Drive: {drive_checkpoint}")
            # Copy to local first
            local_checkpoint = model_dir / 'last.pt'
            shutil.copy(drive_checkpoint, local_checkpoint)
            model = YOLO(str(local_checkpoint))
        else:
            print("⚠ No checkpoint found. Starting fresh training.")
            model_file = f'yolov8{model_size}.pt'
            print(f"Loading pre-trained model: {model_file}")
            model = YOLO(model_file)
    else:
        # Start fresh training
        model_file = f'yolov8{model_size}.pt'
        print(f"Loading pre-trained model: {model_file}")
        model = YOLO(model_file)
    
    print(f"✓ Model loaded: YOLOv8{model_size}")
    
    # Prepare training arguments
    train_args = {
        "data": str(data_yaml_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "name": run_name_dir,
        "patience": patience,
        "save_period": save_period,  # Save checkpoint every N epochs
        "val": val,  # Enable validation (will use valid/images from data.yaml)
        "plots": plots,
        "verbose": True,
        "save": True,  # Always save checkpoints
        "save_json": True,  # Save results JSON
        **kwargs
    }
    
    # Add device if specified
    if device:
        train_args["device"] = device
    
    # CRITICAL: Enable wandb in Ultralytics
    # Ultralytics automatically detects wandb if it's installed and initialized
    # We just need to set the project name - Ultralytics will use wandb if available
    train_args["project"] = project_name
    # NOTE: Do NOT pass wandb=True - it's not a valid argument
    # Ultralytics automatically uses wandb if wandb.init() was called before training
    
    if wandb_initialized:
        print("✓ W&B logging enabled (wandb initialized, Ultralytics will use it automatically)")
    else:
        print("⚠ W&B not manually initialized - Ultralytics may still auto-initialize wandb")
        print("  If wandb is installed, Ultralytics will attempt to initialize it automatically")
    
    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    print(f"  Validation: {'ENABLED' if val else 'DISABLED'}")
    print(f"  Checkpoint saving: Every {save_period} epoch(s)")
    if wandb_initialized:
        print(f"  W&B logging: ENABLED (wandb initialized)")
    else:
        print(f"  W&B logging: Will be auto-enabled by Ultralytics (if wandb is available)")
    print("=" * 60 + "\n")

    # Train the model
    print("Starting model training...")
    print("  Metrics will be logged to W&B in real-time during training")
    print("  Checkpoints will be saved every epoch for Colab safety\n")
    
    # Initialize results to None in case of early interruption
    results = None
    
    try:
        # Train the model
        # W&B metrics are automatically logged by Ultralytics when wandb=True
        results = model.train(**train_args)
        print("\n✓ Training completed successfully!")
        
        # Log final summary to wandb if initialized
        if wandb_initialized and wandb.run:
            # Ultralytics automatically logs all metrics during training
            # But we can log final summary if available
            try:
                # Log model as artifact
                if hasattr(results, 'save_dir') and results.save_dir:
                    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
                    if best_model_path.exists():
                        artifact = wandb.Artifact(
                            f"yolov8{model_size}_ppe_best",
                            type="model",
                            description=f"Best YOLOv8{model_size} model for PPE detection"
                        )
                        artifact.add_file(str(best_model_path))
                        wandb.log_artifact(artifact)
                        print("✓ Best model logged to W&B as artifact")
                        
                        # Also log last checkpoint
                        last_model_path = Path(results.save_dir) / "weights" / "last.pt"
                        if last_model_path.exists():
                            last_artifact = wandb.Artifact(
                                f"yolov8{model_size}_ppe_last",
                                type="model",
                                description=f"Last YOLOv8{model_size} checkpoint"
                            )
                            last_artifact.add_file(str(last_model_path))
                            wandb.log_artifact(last_artifact)
                            print("✓ Last checkpoint logged to W&B as artifact")
            except Exception as e:
                print(f"⚠ Warning: Could not log model artifact: {e}")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user (Ctrl+C)")
        print("  Checkpoint should be saved. You can resume with resume=True")
        if wandb_initialized and wandb.run:
            try:
                wandb.log({"training_status": "interrupted"})
            except:
                pass
        # Set results to None if training was interrupted before completion
        if results is None:
            results = type('obj', (object,), {'save_dir': None})()
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        if wandb_initialized and wandb.run:
            try:
                wandb.log({"training_error": str(e), "training_status": "failed"})
            except:
                pass
        # Set results to None on error
        results = type('obj', (object,), {'save_dir': None})()
        raise
    
    # Save the best model to models directory and backup to Google Drive
    try:
        # The best model is saved by Ultralytics in runs/detect/{name}/weights/best.pt
        source_model_path = project_root / 'runs' / 'detect' / run_name_dir / 'weights' / 'best.pt'
        source_last_path = project_root / 'runs' / 'detect' / run_name_dir / 'weights' / 'last.pt'
        
        if source_model_path.exists():
            # Copy best model to models directory
            target_model_path = model_dir / 'best.pt'
            shutil.copyfile(source_model_path, target_model_path)
            print(f"✓ Best model saved to {target_model_path}")
            
            # Backup to Google Drive if configured
            if drive_backup_dir:
                try:
                    drive_best_path = drive_backup_dir / 'best.pt'
                    shutil.copyfile(source_model_path, drive_best_path)
                    print(f"✓ Best model backed up to Google Drive: {drive_best_path}")
                except Exception as e:
                    print(f"⚠ Warning: Could not backup best model to Drive: {e}")
        else:
            print(f"⚠ Warning: Could not find best model at {source_model_path}")
        
        # Copy last checkpoint (important for resuming)
        if source_last_path.exists():
            target_last_path = model_dir / 'last.pt'
            shutil.copyfile(source_last_path, target_last_path)
            print(f"✓ Last checkpoint saved to {target_last_path}")
            
            # Backup to Google Drive if configured
            if drive_backup_dir:
                try:
                    drive_last_path = drive_backup_dir / 'last.pt'
                    shutil.copyfile(source_last_path, drive_last_path)
                    print(f"✓ Last checkpoint backed up to Google Drive: {drive_last_path}")
                except Exception as e:
                    print(f"⚠ Warning: Could not backup checkpoint to Drive: {e}")
        else:
            print(f"⚠ Warning: Could not find last checkpoint at {source_last_path}")
            
    except Exception as e:
        print(f"⚠ Warning: Error saving/backing up models: {e}")
    
    # Finish wandb run (only if we manually initialized it)
    if wandb_initialized and wandb.run:
        try:
            wandb.finish()
            print("✓ W&B run completed")
        except:
            pass  # Ignore errors when finishing wandb
    
    print("\n" + "=" * 60)
    print("Training Pipeline Complete!")
    print("=" * 60)
    
    # Return results if available, otherwise return None
    return results if results is not None else None


if __name__ == "__main__":
    # Training configuration
    # For Google Colab, recommended settings:
    # - save_period=1 (saves every epoch for safety)
    # - Set google_drive_backup to backup checkpoints
    # - Use resume=True to continue from last checkpoint if interrupted
    
    train_model(
        model_size='m',  # Options: 'n', 's', 'm', 'l', 'x'
        epochs=25,
        imgsz=640,
        batch=8,
        device=None,  # Set to 'cpu', 'cuda', '0', etc. or None for auto
        patience=50,  # Early stopping patience
        save_period=1,  # Save checkpoint every epoch (recommended for Colab)
        val=True,  # Enable validation (uses valid/images from data.yaml)
        resume=False,  # Set to True to resume from last checkpoint
        # google_drive_backup='/content/drive/MyDrive/ppe-models',  # Uncomment for Colab
    )