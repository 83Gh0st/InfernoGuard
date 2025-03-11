# InfernoGuard | AI-Powered Fire Detection System 🔥🚒  
![prediction result](https://github.com/83Gh0st/InfernoGuard/blob/main/predictions/Model/val_batch2_labels.jpg?raw=true)

## **About**  
**InfernoGuard** is an **AI-driven fire detection system** built using **YOLOv5**, optimized for real-time **thermal camera-based fire detection**. It utilizes **ONNX, TensorFlow Lite (TFLite), and TensorFlow.js (TF.js)** for seamless deployment across **edge devices, cloud servers, and web applications**.  

By leveraging **OpenCV’s Deep Neural Network (DNN) module**, InfernoGuard can detect **fire outbreaks in images, videos, and live streams**, making it an ideal solution for **early fire detection in industrial plants, forests, warehouses, and residential buildings**.  

---

## **🚀 Key Features**  
✅ **Thermal Camera-Based Fire Detection** – Works in low visibility conditions.  
✅ **Optimized for Multiple Platforms**:  
   - **ONNX** (High-performance inference on CPU/GPU)  
   - **TFLite** (Optimized for mobile, edge, and embedded systems)  
   - **TF.js** (Runs in a browser for web-based monitoring)  
✅ **Real-Time Performance** – Works with **CCTV, drones, and IoT-enabled cameras**.  
✅ **Edge AI Deployment** – Compatible with **Raspberry Pi, NVIDIA Jetson, and cloud-based solutions**.  

---

## **🎯 Use Cases**  
🔹 **Early fire detection in industrial zones, warehouses, and factories.**  
🔹 **Forest fire monitoring using drones.**  
🔹 **Smart surveillance for residential and commercial buildings.**  
🔹 **IoT integration for automatic fire alarm activation.**  

---

## **🖥️ Demo**  
### **Run on Image Input**  
```python
import cv2
from yolo_predictions import YOLO_Pred

# Load Model
yolo = YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')

# Load Image
img = cv2.imread('test_image.jpg')

# Perform Detection
img_pred = yolo.predictions(img)

# Display Results
cv2.imshow('Fire Detection', img_pred)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### **Run Real-Time Detection on Webcam**  
```python
cap = cv2.VideoCapture(0)  # Use webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pred_image = yolo.predictions(frame)
    cv2.imshow('Fire Detection', pred_image)

    if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
```

---

## **🛠️ Model Variants & Deployment Options**  

| **Model Format** | **Use Case** |
|------------------|-------------|
| **ONNX**  | High-performance inference on CPU/GPU devices |
| **TFLite**  | Optimized for mobile, edge, and embedded systems |
| **TF.js**  | Runs in a browser for real-time monitoring |

---

## **📊 Model Performance & Results**  

✅ **mAP (Mean Average Precision):** **92.4%**  
✅ **Precision:** **94.1%**  
✅ **Recall:** **89.8%**  

📈 **Training Metrics (Loss, Accuracy, Confusion Matrix) are available in the `results/` folder.**  

---

## **📂 Project Structure**  

```
├── Model/
│   ├── weights/
│   │   ├── best.onnx  # ONNX Model
│   │   ├── best.tflite  # TFLite Model
│   │   ├── best_web_model/  # TF.js Model
│   ├── results/  # Training graphs and evaluation results
│
├── predictions/
│   ├── detect.py  # Image/Video/Webcam detection script
│   ├── yolo_predictions.py  # YOLO inference class
│   ├── utils.py  # Helper functions
│
├── dataset/
│   ├── images/
│   ├── labels/
│── data.yaml  # Dataset configuration
│
├── README.md  # Project Documentation
```

---

## **🚀 Installation & Setup**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/83Gh0st/InfernoGuard.git
cd InfernoGuard
```

### **2️⃣ Install Dependencies**  

### **3️⃣ Run Object Detection**  

🔹 **On Images**  
```bash
python3 detect.py --source test_image.jpg
```
🔹 **On Webcam**  
```bash
python3 detect.py --source 0
```

---

## **📦 Model Conversion & Deployment**  

### **Convert PyTorch Model to ONNX**  
```python
import torch

model = torch.load('best.pt', map_location='cpu')
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy_input, "best.onnx", opset_version=11)
```

### **Convert ONNX to TensorFlow Lite**  
```bash
onnx-tf convert -i best.onnx -o best.pb
tflite_convert --saved_model_dir=best.pb --output_file=best.tflite
```

### **Convert TensorFlow Model to TF.js**  
```bash
tensorflowjs_converter --input_format=tf_saved_model best.pb best_web_model/
```

---

## **📌 Deployment on Web Using TensorFlow.js**  

1️⃣ **Copy the `best_web_model/` to your web server.**  
2️⃣ **Load the model in JavaScript:**  

```javascript
const model = await tf.loadGraphModel('best_web_model/model.json');
const img = tf.browser.fromPixels(document.getElementById('input_image'));
const predictions = model.predict(img);
```

---

## **🛠️ Future Improvements**  

🔹 **IoT Integration for automated fire alarm triggers.**  
🔹 **Deploy as a cloud-based AI API for smart surveillance.**  
🔹 **Expand dataset to include more fire intensity variations.**  
🔹 **Optimize for real-time edge AI applications (e.g., NVIDIA Jetson Nano).**  

---

## **📜 License**  
This project is licensed under the **MIT License** – Free to use, modify, and distribute.  

---

## **👨‍💻 Author**  
Developed by **@83Gh0st** 🔥  
💬 **Contact:** [GitHub](https://github.com/83Gh0st)  

🔥 **Star this repo if you found it useful!** ⭐  

