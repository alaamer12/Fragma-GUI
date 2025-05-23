# Fragma-GUI

A modular Gradio-based interface for experimenting with classical ML and deep learning models on text data. Designed to help you interact with models through a clean UI and easily switch between different backends.


## Project Structure

```
.
├── README.md             # Project documentation
├── app.py                # 🔁 Main file to switch between UIs
├── ml_ui.py              # 🧠 ML model UI (already working)
├── dl_ui.py              # 🧠 DL model UI (placeholder for now)
├── requirements.txt      # Project dependencies
├── models/               # Models foleder 
│   └── fragma_ml.pkl/    # Fragment ML classifier        
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alaamer12/fragma-gui.git
   cd fragma-gui
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate      # On Windows
    source .venv/bin/activate   # On macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀Usage

### Running the app
```
 python app.py
```



## ✨ Features
```
✅ Modular UI using Gradio

✅ Pretrained classical ML model (RandomForestClassifier)

🔄 Easy switching between ML and DL models

🔧 Extendable architecture for future deep learning integration

🧪 Regex-based feature extraction for NLP preprocessing

✅ Classical ML model with regex features

🔜 Integration of a deep learning text classifier (e.g., BERT, LSTM)
```





## 🧪 Planned Features
```
🧠 Model explanation tools (like SHAP)

💾 Option to upload text files for batch prediction
```
