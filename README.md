# Multi-Disease Chest X-ray Classifier with Explainable AI

An academic web application that uses deep learning to predict multiple possible findings from a chest X-ray and provides visual explanations using Grad-CAM.

## Project Overview

This project combines a machine learning model with a web interface to explore automated chest X-ray classification. Users can upload an X-ray image, view predicted disease probabilities, and visualize relevant image regions using Grad-CAM.

The application is divided into three main components:

- **Frontend:** React and Vite provide the user interface for uploading images and viewing results.
- **Backend:** FastAPI handles image requests and connects the frontend to the prediction and explainability services.
- **Machine Learning:** PyTorch-based code supports model training, inference, evaluation, and Grad-CAM explainability.

## Key Features

- Upload a chest X-ray image through the web interface.
- Predict multiple possible chest X-ray findings.
- Display prediction probabilities for supported labels.
- Generate Grad-CAM heatmaps to help visualize image regions influencing model predictions.
- Separate the frontend, backend, and machine learning components for easier development and maintenance.

## Technologies Used

| Component | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| Backend | Python, FastAPI |
| Machine Learning | PyTorch, Torchvision |
| Explainable AI | Grad-CAM |
| Version Control | Git, GitHub |

## Project Structure

```text
multi-disease-chest-xray-classifier/
├── backend/
│   └── app/
│       ├── api/
│       ├── services/
│       └── main.py
├── frontend/
│   ├── public/
│   └── src/
├── ml/
│   ├── configs/
│   ├── data/
│   ├── checkpoints/
│   └── src/
│       ├── datasets/
│       ├── evaluation/
│       ├── explainability/
│       ├── inference/
│       ├── models/
│       ├── training/
│       └── utils/
├── docs/
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

Install the following software before running the application:

- Python 3.10 or another version compatible with the project's dependencies.
- Node.js and npm.
- Git.
- A compatible trained model checkpoint.

### 1. Clone the Repository

```bash
git clone https://github.com/Pri0713/multi-disease-chest-xray-classifier.git
cd multi-disease-chest-xray-classifier
```

### 2. Set Up the Python Environment

From the project root, create a virtual environment if you do not already have one:

```bash
python -m venv ml/.venv
```

Activate it on Windows PowerShell:

```powershell
.\ml\.venv\Scripts\Activate.ps1
```

Install the dependencies listed in the machine learning requirements file:

```bash
python -m pip install -r ml/requirements.txt
```

If the backend requires additional packages not included in that file, install the backend dependencies specified by the project.

### 3. Obtain the Trained Model

The trained model checkpoint is not included in the Git repository because model files are excluded by `.gitignore`.

Obtain the project's compatible checkpoint from the project team and place it at the path expected by the prediction service. For the current local setup, this may be:

```text
ml/checkpoints/best_model.pth
```

Confirm the required filename, architecture, and checkpoint path in the model and prediction code before running the application.

### 4. Start the Backend

From the project root, run:

```powershell
& ".\ml\.venv\Scripts\python.exe" -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

The backend should be available at:

- API: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs

Keep this terminal running.

### 5. Start the Frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in the terminal. In the current development setup, this is expected to be:

http://localhost:5174

Keep both the backend and frontend running while using the application.

## How to Use the Application

1. Open the frontend in your browser.
2. Select a chest X-ray image using the upload interface.
3. Submit the image for analysis.
4. Review the predicted findings and their probabilities.
5. View the Grad-CAM visualization, when available, to explore the regions influencing a prediction.

The available results depend on the model, its configuration, and the image submitted.

## Explainable AI with Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) produces a heatmap highlighting image regions that contribute to a model's prediction for a selected class.

In this project, Grad-CAM provides a visual explanation alongside the predicted findings. The heatmap is an interpretation of model behavior; it does not prove that a disease is present in a highlighted region.

## Dataset and Model Evaluation

The machine learning component includes code for dataset handling, training, inference, and evaluation.

Dataset files and trained model weights may need to be obtained separately. They are not necessarily included in the public repository.

Model performance should be assessed using appropriate multi-label classification metrics, including precision, recall, F1-score, and AUROC. Performance depends on the evaluation data, decision thresholds, class imbalance, and consistency between training and inference preprocessing.

## Limitations and Disclaimer

- This project is developed for academic and educational purposes.
- Predictions may contain false positives and false negatives.
- Grad-CAM visualizations are approximate explanations and should not be treated as clinical evidence.
- The system has not been established as a clinically validated diagnostic tool.
- Results must not be used as a substitute for evaluation by a qualified healthcare professional.

## Future Improvements

- Improve model evaluation and class-specific performance.
- Enhance user feedback and error handling.
- Improve reproducibility of model training and inference.
- Expand testing and documentation.
- Explore further explainability and model validation methods.

## Contributors

This project was developed collaboratively as a semester project. Contributors worked on the frontend, backend, machine learning, and explainability components.

## License

See the [LICENSE](LICENSE) file for licensing information.
