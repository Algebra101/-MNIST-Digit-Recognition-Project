import gradio as gr
import joblib
import numpy as np
from PIL import Image, ImageOps


# Load model and scaler
loaded_model = joblib.load("mnist_rf_model.joblib")
scaler = joblib.load("scaler.joblib")

def predict_digit(image):
    try:
        if isinstance(image, dict):
            layers = image.get("layers", [])
            if not layers:
                return "Please draw a digit before submitting."
            image_array = np.array(layers[0])
        else:
            image_array = image

        img = Image.fromarray(image_array).convert("L")
        img = ImageOps.invert(img)
        img = img.resize((28, 28))

        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 784)

        # 🔧 Scale using the same scaler used during training
        img_array_scaled = scaler.transform(img_array)

        prediction = loaded_model.predict(img_array_scaled)
        return int(prediction[0])

    except Exception as e:
        print("🔥 ERROR:", e)
        return f"Error: {e}"


# Define the Gradio interface
# Gradio interface (no shape, no dict access, pure array)
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(scale=10),  # Resizable canvas
    outputs=gr.Label(num_top_classes=10),  # Single output for digit prediction
    live=True,  # Enable live prediction
    title="Digit Recognition App",
    description="Draw a digit (0–9) and get an instant prediction!"
)

if __name__ == "__main__":
    interface.launch(share=True)
