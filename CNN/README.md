Apple vs Banana Image Classifier (TensorFlow + Colab Webcam)
This project builds a binary image classifier using a Convolutional Neural Network (CNN) to distinguish between apples and bananas. It leverages Google Colab for training and real-time prediction using the webcam.

🚀 Features
Trains a CNN model using ImageDataGenerator for augmentation and normalization.

Dataset is loaded from Google Drive.

Webcam integration for live image capture and prediction.

Model saves and reloads with .h5 format.

Displays processed input and prediction result.

🧠 Tech Stack
TensorFlow / Keras (for CNN model building and training)

Google Colab (notebook execution and webcam support)

OpenCV, NumPy, Matplotlib (image processing and visualization)

📂 Project Structure
Dataset: Stored in Google Drive in two folders (apples, bananas)

Model: Trained CNN with Conv2D, MaxPooling, Dense layers

Webcam Prediction:

Capture photo

Preprocess (resize, normalize)

Predict with trained model

✅ Run the Model
Upload your dataset (apples, bananas) to Google Drive.

Mount drive in Colab.

Train the CNN.

Capture a photo using webcam.

The model will predict whether the image is an apple or banana.

📌 Output Example
rust
Copy
Edit
Raw prediction: [[0.968]]
It's a banana!
💾 Model
Saved as: apple_banana_classifier.h5
Recommended format: my_model.keras

