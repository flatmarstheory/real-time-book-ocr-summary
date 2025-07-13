# Real-Time Book OCR + Summary Generator 📚🧠

This project is a real-time computer vision + AI application that detects book covers via webcam, extracts their cover text using OCR, and then generates a short, 100-word summary using a locally hosted LLM through [Ollama](https://ollama.com/). The app features a responsive, live GUI with video feed, face privacy, and AI output.

<img width="1184" height="715" alt="image" src="https://github.com/user-attachments/assets/cb5af90e-39b7-4eea-a24f-babf65cdba12" />

🎥 **YouTube Demo:** [https://youtu.be/cJKqo_BKpWE](https://youtu.be/cJKqo_BKpWE)

---

## 🧠 What it does

- Detects book covers in a live video stream using YOLOv5
- Extracts cover text using Tesseract OCR (with optional EasyOCR fallback)
- Buffers 10 lines of meaningful text
- Sends the text to a local LLM (like LLaMA 2 or Phi3 via Ollama) for summarization
- Displays a live summary (max 100 words) in a side-by-side GUI
- Protects privacy by pixelating detected faces
- Resets after each summary so it's ready to scan the next book

---

## 🖥️ Technologies Used

- Python + OpenCV + Tkinter
- Ultralytics YOLOv5 (via `ultralytics` package)
- Tesseract OCR (`pytesseract`)
- EasyOCR (optional)
- Ollama for LLM inference
- PIL (Pillow) for image conversion

---

## 🔧 Setup Instructions

> These steps are for **Windows 11**, but the logic works on Linux/macOS too.

### 1. Clone or Download the Repo
```bash
git clone https://github.com/your-username/real-time-book-ocr-summary.git
cd real-time-book-ocr-summary
````

### 2. Install Dependencies

Install Python packages:

```bash
pip install ultralytics opencv-python-headless pytesseract easyocr pillow
```

Install Tesseract OCR:

* Download from: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
* Make sure to install it to:
  `C:\Program Files\Tesseract-OCR`
* Add to PATH or update this line in code:

  ```python
  pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
  ```

### 3. Install Ollama

Install from: [https://ollama.com/download](https://ollama.com/download)
Then pull your model:

```bash
ollama pull llama2
```

*(Or try `phi3` for faster inference.)*

### 4. Run the App

```bash
python object_detection_gui.py
```

---

## ✅ Features Recap

* Live detection of books via webcam
* OCR-based text extraction
* LLM-based summarization via Ollama
* In-app loading animation while generating summaries
* Privacy-preserving face pixelation
* Continuous scanning without restarting
* 9:16 GUI layout (video left, summary right)

---

## 📺 Demo

[![Watch the demo video](https://img.youtube.com/vi/cJKqo_BKpWE/hqdefault.jpg)](https://youtu.be/cJKqo_BKpWE)

---

## 🙌 Credits

* Ultralytics for YOLOv5
* Tesseract & EasyOCR for OCR
* Ollama for local LLM serving
* You — for trying this out!

---

## 📄 License

MIT License — use freely, but attribution appreciated.
