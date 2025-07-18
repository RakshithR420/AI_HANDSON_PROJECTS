# 🍎🍌 Apple vs Banana Image Classifier

This project is a binary image classifier built using **TensorFlow** and **Keras** that distinguishes between apples and bananas. It supports real-time image capture and prediction using a webcam inside **Google Colab**.

---

## 🚀 Features

- CNN model using Conv2D, MaxPooling, Flatten, and Dense layers.
- Real-time webcam integration for capturing and classifying images.
- Data augmentation and preprocessing with `ImageDataGenerator`.
- Easy dataset loading from Google Drive.
- Saves and reloads the trained model (`.h5` format).

---

## 🧠 Tech Stack

- **TensorFlow / Keras**
- **Google Colab**
- **OpenCV**, **NumPy**, **Matplotlib**

---

## 📂 Dataset Structure

dataset/
├── apples/
│ ├── apple1.jpg
│ └── ...
└── bananas/
├── banana1.jpg
└── ...


> The dataset is stored in your Google Drive and loaded via `flow_from_directory`.

---

## 🧪 Model Architecture

- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Flatten → Dense → Sigmoid

---

## 🧾 Usage Steps

1. Mount your Google Drive in Colab:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. Load dataset using:
    ```python
    ImageDataGenerator(rescale=1./255, validation_split=0.2)
    ```

3. Train the model:
    ```python
    model.fit(train_generator, epochs=15, validation_data=validation_generator)
    ```

4. Save the model:
    ```python
    model.save("apple_banana_classifier.h5")
    ```

5. Capture image from webcam:
    ```python
    filename = take_photo()
    ```

6. Preprocess and predict:
    ```python
    prediction = model.predict(img_input)
    ```

---

## ✅ Sample Output

Raw prediction: [[0.968]]
It's a banana!

yaml
Copy
Edit

---

## 💾 Model Info

- File saved: `apple_banana_classifier.h5`
- You can also use: `model.save("model.keras")` for newer format.

---

## 📸 Live Demo

Webcam capture is implemented directly in Colab using JavaScript and Python integration. The captured image is resized and passed into the CNN for prediction.

---

## 📌 Notes

- Ensure webcam permissions are granted in Colab.
- Recommended input image size: **128x128x3**.
- For best accuracy, use a well-lit environment when capturing photos.

---
