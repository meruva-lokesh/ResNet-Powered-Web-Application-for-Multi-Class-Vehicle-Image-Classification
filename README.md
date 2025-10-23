Here's a clean and well-structured README file for your Vehicle Classification System:

---

# 🚗 Vehicle Classification System

## 📝 Description
A deep learning-based web application that classifies different types of vehicles using the **ResNet50** architecture. The system provides real-time classification with confidence scores through an intuitive web interface.

## ✨ Features
- ✅ **Real-time vehicle classification**
- ✅ **Interactive web interface with drag-and-drop functionality**
- ✅ **Built on ResNet50 architecture**
- ✅ **Image preprocessing and augmentation**
- ✅ **Confidence score display**
- ✅ **Supports multiple image formats (PNG, JPG, JPEG)**
- ✅ **Modern UI with glassmorphism design**

## 🛠️ Technologies Used

### Backend:
- Python 3.x
- TensorFlow
- Flask
- Joblib
- NumPy

### Frontend:
- HTML5
- CSS3
- JavaScript
- Modern **glassmorphism** UI

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vehicle-classifier.git
   cd vehicle-classifier
   ```

2. **Create and activate a virtual environment**
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Upload an image of a vehicle and get instant classification results!**

## 📋 Project Structure

```
vehicle-classifier/
│
├── app.py                 # Flask application
├── model_training.py      # Model training script
├── templates/
│   └── index.html         # Frontend interface
├── static/
│   └── uploads/          # Folder for uploaded images
├── vehicle_classifier.h5  # Trained model
└── class_labels.pkl       # Class labels
```

## 🔧 Model Architecture

- **Base Model**: ResNet50  
- **Input Size**: 224x224 pixels  
- **Output**: Multiple vehicle classes  

### Training Parameters:
- **Batch Size**: 32  
- **Epochs**: 20  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  

## 🔑 Requirements

Ensure you have the following dependencies installed:

```
tensorflow>=2.0.0
flask>=2.0.0
numpy>=1.19.2
pillow>=8.0.0
joblib>=1.0.0
flask-cors>=3.0.0
```


---

