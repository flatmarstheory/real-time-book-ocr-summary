import re
import cv2
import pytesseract
import subprocess
import threading
from ultralytics import YOLO
from collections import deque
from tkinter import *
from PIL import Image, ImageTk

# ── CONFIG ──
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OLLAMA_MODEL = "llama2"
BUF_SIZE = 10
TIMEOUT_SECONDS = 90

text_buffer = deque(maxlen=BUF_SIZE)
current_summary = ""
cap = None
closing = False
video_size = (640, 480)

# ── Models ──
model = YOLO('yolov5su.pt')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    import easyocr
    ocr_reader = easyocr.Reader(['en'])
    use_easyocr = True
except:
    use_easyocr = False

# ── GUI ──
root = Tk()
root.title("Live Book OCR + Summary")
root.geometry("1200x600")
root.configure(bg="black")

main_frame = Frame(root, bg="black")
main_frame.pack(fill=BOTH, expand=True)

video_label = Label(main_frame, bg="black")
video_label.pack(side=LEFT, padx=10, pady=10)

summary_label = Label(main_frame, text="Waiting for summary...",
                      fg="cyan", bg="black",
                      font=("Courier", 14, "bold"),
                      justify=LEFT, wraplength=500)
summary_label.pack(side=RIGHT, padx=10, pady=10, fill=BOTH, expand=True)

overlay_frame = Frame(root, bg="yellow", bd=5, relief=RIDGE)
overlay_label = Label(overlay_frame, text="PLEASE WAIT...",
                      font=("Arial", 24, "bold"),
                      bg="yellow", fg="black", padx=20, pady=20)
overlay_label.pack()

# ── OCR ──
def clean_and_threshold(gray):
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh, cv2.bitwise_not(thresh)

def ocr_confidence(img):
    data = pytesseract.image_to_data(
        img, output_type=pytesseract.Output.DICT,
        config='--oem 3 --psm 6', lang='eng')
    words = [re.sub(r'[^A-Za-z ]+', '', t).strip()
             for t, c in zip(data['text'], data['conf']) if int(c) >= 60]
    return ' '.join([w for w in words if len(w) > 1])

def extract_text(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    t1, t2 = clean_and_threshold(gray)
    r1 = ocr_confidence(t1)
    r2 = ocr_confidence(t2)
    return r1 if len(r1) > len(r2) else r2

def extract_text_easyocr(roi):
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    res = ocr_reader.readtext(rgb, detail=0)
    clean = [re.sub(r'[^A-Za-z ]+', '', r).strip() for r in res if len(r) > 1]
    return ' '.join(clean)

# ── LLM ──
def summarize_with_ollama(texts):
    overlay_frame.place(relx=0.5, rely=0.5, anchor="center")
    overlay_label.config(text="PLEASE WAIT...")
    root.update()

    prompt = (
        "You are a helpful assistant. "
        "Given the following extracted texts from a book cover:\n\n"
        + "\n".join(f"- {t}" for t in texts)
        + "\n\nProvide a brief summary of what this book might be about. Limit your summary to 100 words."
    )

    try:
        proc = subprocess.Popen(
            ["ollama", "run", OLLAMA_MODEL],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, err = proc.communicate(input=prompt, timeout=TIMEOUT_SECONDS)
        if proc.returncode != 0:
            return f"[Ollama error: {err.strip()}]"
        return trim_to_100_words(out.strip())
    except Exception as e:
        return f"[Ollama exception: {e}]"
    finally:
        overlay_frame.place_forget()

def trim_to_100_words(text):
    words = text.split()
    return " ".join(words[:100]) + ("..." if len(words) > 100 else "")

def animate_summary(text):
    def type_step(index=0):
        if index <= len(text):
            summary_label.config(text=text[:index])
            root.after(30, lambda: type_step(index + 1))
    type_step()

# ── Preload ──
def preload_ollama_model():
    try:
        subprocess.run(["ollama", "run", OLLAMA_MODEL],
                       input="hello\n", text=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       timeout=TIMEOUT_SECONDS)
        print(f"✅ Model '{OLLAMA_MODEL}' preloaded.")
    except Exception as e:
        print(f"⚠️ Preload failed: {e}")

# ── Detection ──
def detection_loop():
    global current_summary, cap, video_size
    cap = cv2.VideoCapture(0)
    ret, sample_frame = cap.read()
    if ret:
        video_size = (sample_frame.shape[1], sample_frame.shape[0])

    while not closing:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(source=frame, conf=0.5, verbose=False)
        detections = results[0].boxes
        annotated = frame.copy()

        for box in detections:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == 'book':
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                text = extract_text(roi)
                if use_easyocr and not text:
                    text = extract_text_easyocr(roi)

                if text and len(text) >= 3:
                    text_buffer.append(text)
                    if len(text_buffer) == BUF_SIZE:
                        current_summary = summarize_with_ollama(list(text_buffer))
                        text_buffer.clear()
                        root.after(0, lambda: animate_summary(current_summary))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, text[:25], (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            elif label == 'person':
                gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                for (fx, fy, fw, fh) in faces:
                    fx1, fy1 = x1 + fx, y1 + fy
                    fx2, fy2 = fx1 + fw, fy1 + fh
                    face = annotated[fy1:fy2, fx1:fx2]
                    if face.size == 0:
                        continue
                    temp = cv2.resize(face, (10, 10))
                    pix = cv2.resize(temp, (fw, fh), interpolation=cv2.INTER_NEAREST)
                    annotated[fy1:fy2, fx1:fx2] = pix

        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=image)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cap:
        cap.release()

def on_closing():
    global closing
    closing = True
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
threading.Thread(target=preload_ollama_model, daemon=True).start()
threading.Thread(target=detection_loop, daemon=True).start()
root.mainloop()
