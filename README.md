# Smart OCR Application

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- CUDA installed (for GPU acceleration)
- Git
- Poppler (required for PDF processing - see installation instructions below)

### Poppler Installation

#### Windows Users:
1. Download Poppler for Windows from: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract the downloaded file
3. Add the `bin` folder path to your system's PATH environment variable:
   - Right-click on 'This PC' or 'My Computer'
   - Click 'Properties'
   - Click 'Advanced system settings'
   - Click 'Environment Variables'
   - Under 'System Variables', find and select 'Path'
   - Click 'Edit'
   - Click 'New'
   - Add the path to your Poppler bin folder (e.g., `C:\Program Files\poppler-xx\bin`)
   - Click 'OK' on all windows

#### Linux Users:
```bash
sudo apt install -y poppler-utils
```

### Installation in 3 Simple Steps

1. **Clone the repository**
```bash
git clone https://github.com/ha684/Demo_OCR.git
cd Demo_OCR
```

2. **Set up Python environment**
```bash
# Create virtual environment
python -m venv env

# Activate environment
# For Windows:
env\Scripts\activate
# For Linux/Mac:
source env/bin/activate
```

3. **Install dependencies**
```bash
# Install using setup.py (recommended)
pip install -e .

```

## 🎯 Features

- **Layout Detection**: YOLO-based document layout analysis
- **OCR Options**: 
  - PaddleOCR (Fast, general-purpose)
  - Vision Language Model (High accuracy)
- **File Support**: 
  - Images (JPG, PNG, JPEG)
  - PDF documents
- **User Interface**: Clean Streamlit interface
- **Output**: Save extracted text as TXT or DOCX

## 🎮 How to Run

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and go to:
```
http://localhost:8501
```

## 💡 Usage Tips

1. **Upload Files**: Use the sidebar to upload your documents
2. **Choose OCR Method**: 
   - PaddleOCR: Best for simple documents and quick results
   - Vision Language Model: Better for complex layouts and accuracy
3. **View Results**: See processed images and extracted text in real-time
4. **Download**: Save the extracted text in your preferred format

## 📝 Note

- First-time setup might take a few minutes to download models
- GPU is recommended for better performance
- For large PDF files, processing might take longer
- Make sure Poppler is properly installed for PDF processing

## 🎥 Demo

[Demo video coming soon]

## 🤝 Support

For issues or questions, please:
1. Open a GitHub issue
2. Contact: phanha6844@gmail.com

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.