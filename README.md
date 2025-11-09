# PPE Detection Pipeline

> **⚠️ Note**: This project is currently under active development. New features and improvements are being added regularly. The documentation and codebase may be updated frequently. Please check back for the latest updates!

An end-to-end MLOps project for building a real-time Personal Protective Equipment (PPE) detection system using YOLOv8. This project demonstrates a complete ML pipeline from data versioning to model deployment.

## 🎯 Overview

This project implements a comprehensive PPE detection system that can identify various safety equipment items in images/videos, including:
- Hardhats, Gloves, Goggles, Masks, Safety Vests
- Detection of missing PPE (NO-Hardhat, NO-Gloves, etc.)
- Person detection and fall detection
- Safety cones and ladders

The system is designed with MLOps best practices, including data versioning, experiment tracking, model deployment, and a user-friendly dashboard.

## ✨ Features

- **Data Versioning**: Track large datasets using DVC with Backblaze B2 storage
- **Model Training**: Train YOLOv8 models with comprehensive experiment tracking
- **Experiment Tracking**: Log all training metrics to Weights & Biases (W&B)
- **Model Deployment**: REST API using FastAPI for production inference
- **Containerization**: Docker support for easy deployment
- **Interactive Dashboard**: Streamlit UI for testing and visualization
- **Colab Support**: Train models in Google Colab with checkpoint resumption
- **Checkpoint Management**: Automatic checkpointing and Google Drive backup

## 🛠 Tech Stack

- **ML Model**: YOLOv8m (Ultralytics)
- **ML Framework**: PyTorch
- **Data Versioning**: DVC (Data Version Control)
- **Remote Storage**: Backblaze B2 (S3-compatible)
- **Experiment Tracking**: Weights & Biases (W&B)
- **Backend API**: FastAPI
- **Containerization**: Docker
- **Demo UI**: Streamlit
- **Language**: Python 3.8+

## 📁 Project Structure

```
ppe-detection-pipeline/
├── api/                    # FastAPI backend
│   ├── main.py            # API endpoints
│   ├── requirements.txt   # API dependencies
│   └── Dockerfile         # Docker configuration
├── dashboard/             # Streamlit dashboard
│   ├── app.py            # Dashboard application
│   └── requirements.txt  # Dashboard dependencies
├── data/                  # Dataset directory
│   └── ppe-dataset/      # PPE detection dataset
│       ├── data.yaml     # Dataset configuration
│       ├── train/        # Training images and labels
│       ├── valid/        # Validation images and labels
│       └── test/         # Test images and labels
├── models/               # Trained model weights
├── runs/                 # Training runs and results
├── scripts/              # Training scripts
│   └── train.py         # Main training script
├── ppe_train_in_colab.ipynb  # Colab training notebook
├── requirements.txt      # Main dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- DVC installed (`pip install dvc[s3]`)
- W&B account (for experiment tracking)
- Backblaze B2 account (for data storage)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ppe-detection-pipeline.git
   cd ppe-detection-pipeline
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r api/requirements.txt
   pip install -r dashboard/requirements.txt
   ```

4. **Set up DVC remote storage**
   ```bash
   # Configure Backblaze B2 credentials
   dvc remote modify --local myremote access_key_id YOUR_B2_ACCESS_KEY
   dvc remote modify --local myremote secret_access_key YOUR_B2_SECRET_KEY
   ```

5. **Download the dataset**
   ```bash
   dvc pull
   ```

## 📊 Dataset

The project uses a PPE detection dataset with 14 classes:
- Fall-Detected
- Gloves, NO-Gloves
- Goggles, NO-Goggles
- Hardhat, NO-Hardhat
- Mask, NO-Mask
- NO-Safety Vest
- Person
- Safety Cone
- Safety Vest
- Ladder

The dataset is versioned using DVC and stored on Backblaze B2. To download:
```bash
dvc pull
```

## 🎓 Training

### Local Training

1. **Set up W&B**
   ```bash
   export WANDB_API_KEY=your_wandb_api_key
   wandb login
   ```

2. **Run training**
   ```bash
   python scripts/train.py
   ```

3. **Customize training parameters**
   ```python
   from scripts.train import train_model
   
   train_model(
       model_size='m',      # 'n', 's', 'm', 'l', 'x'
       epochs=25,
       imgsz=640,
       batch=8,
       device='cuda',       # or 'cpu'
       save_period=1,       # Save checkpoint every epoch
       val=True,            # Enable validation
   )
   ```

### Google Colab Training

1. **Open the Colab notebook**
   - Open `ppe_train_in_colab.ipynb` in Google Colab
   - Or use the Colab badge in the notebook

2. **Configure secrets**
   - Set up Colab secrets for:
     - `WANDB_API_KEY`
     - `B2_ACCESS_KEY_ID`
     - `B2_SECRET_ACCESS_KEY`
     - `GITHUB_USER`, `GITHUB_EMAIL`, `GITHUB_TOKEN`

3. **Run training with Google Drive backup**
   ```python
   from scripts.train import train_model
   
   train_model(
       model_size='m',
       epochs=25,
       batch=8,
       save_period=1,  # Save every epoch for Colab safety
       google_drive_backup='/content/drive/MyDrive/ppe-models',
   )
   ```

4. **Resume training if interrupted**
   ```python
   train_model(
       resume=True,  # Resume from last checkpoint
       google_drive_backup='/content/drive/MyDrive/ppe-models',
   )
   ```

### Training Features

- **Automatic checkpointing**: Saves model checkpoints every epoch
- **W&B integration**: Logs all metrics (loss, mAP, precision, recall) in real-time
- **Validation**: Automatic validation after each epoch
- **Resume support**: Continue training from the last checkpoint
- **Google Drive backup**: Backup checkpoints to Google Drive (Colab)
- **Early stopping**: Stops training if validation metrics don't improve

### Monitoring Training

- **W&B Dashboard**: View real-time metrics at https://wandb.ai
- **Local logs**: Check `runs/detect/` for training outputs
- **Checkpoints**: Saved in `runs/detect/yolov8m_ppe_detection/weights/`

## 🚀 API Deployment

### Local Deployment

1. **Start the API server**
   ```bash
   cd api
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Test the API**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@path/to/image.jpg"
   ```

### Docker Deployment

1. **Build the Docker image**
   ```bash
   cd api
   docker build -t ppe-detection-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 -v $(pwd)/../models:/app/models ppe-detection-api
   ```

3. **Access the API**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs

### API Endpoints

- `POST /predict`: Predict PPE in an uploaded image
- `GET /health`: Health check endpoint
- `GET /model/info`: Get model information
- `GET /docs`: Interactive API documentation

## 🎨 Dashboard

### Running the Dashboard

1. **Start the Streamlit app**
   ```bash
   cd dashboard
   streamlit run app.py
   ```

2. **Access the dashboard**
   - Open http://localhost:8501 in your browser

### Dashboard Features

- Upload images for PPE detection
- View detection results with bounding boxes
- Display confidence scores and class labels
- Statistics and metrics visualization
- Batch processing support

## 📈 Experiment Tracking

All training runs are automatically logged to Weights & Biases with:
- Training and validation loss curves
- mAP (mean Average Precision) metrics
- Precision and recall curves
- Confusion matrices
- System metrics (GPU usage, memory)
- Model artifacts (best and last checkpoints)

View your runs at: https://wandb.ai/your-username/ppe-detection-pipeline

## 🔧 Configuration

### Training Configuration

Edit `scripts/train.py` to customize:
- Model size (nano, small, medium, large, xlarge)
- Number of epochs
- Batch size
- Image size
- Learning rate
- Data augmentation

### Dataset Configuration

Edit `data/ppe-dataset/data.yaml` to:
- Update dataset paths
- Modify class names
- Adjust number of classes

### API Configuration

Edit `api/main.py` to:
- Change model path
- Modify confidence thresholds
- Add custom endpoints

## 🐛 Troubleshooting

### W&B Authentication Issues

If you encounter W&B authentication errors:
1. Verify your API key: `echo $WANDB_API_KEY`
2. Try offline mode: The script automatically falls back to offline mode
3. Sync offline runs later: `wandb sync <run_directory>`

### Colab Disconnections

To prevent Colab runtime disconnections:
1. Use `save_period=1` to save checkpoints every epoch
2. Enable Google Drive backup
3. Use `resume=True` to continue from the last checkpoint
4. Monitor training via W&B dashboard

### Checkpoint Not Found

If checkpoints aren't found:
1. Check `runs/detect/` directory
2. Verify training completed at least one epoch
3. Check Google Drive backup location (if using Colab)

## 📝 License

This project uses a dataset licensed under CC BY 4.0 from Roboflow Universe.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Roboflow](https://roboflow.com/) for the PPE dataset
- [Weights & Biases](https://wandb.ai/) for experiment tracking
- [DVC](https://dvc.org/) for data versioning

## 📚 Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [DVC Documentation](https://dvc.org/doc)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Note**: This project is for educational and research purposes. Ensure compliance with safety regulations when deploying in production environments.

