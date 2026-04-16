# Mediseena: Vision-Based Ingestion System 💊

Mediseena is an AI-powered prototype designed to verify medication ingestion using computer vision. It integrates **YOLOv8** for pill detection and **MediaPipe** for hand/mouth tracking.

## 🛠️ Installation & Environment Setup

Since you are using Windows PowerShell, follow these steps:

### 1. Create a Virtual Environment

python -m venv venv

### 2. Activate the Environment
If you receive an error about "scripts being disabled," run the first command, then the second:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1

### 3. Install Requirements
Ensure your requirements.txt is in the same folder, then run:

pip install -r requirements.txt

### 🚀 How to Run the Application
To launch the Streamlit interface:

python -m streamlit run app.py