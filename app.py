import gradio as gr
import joblib
import numpy as np
from PIL import Image, ImageOps

loaded_model = joblib.load("mnist_model.joblib")

def predict_digit(image):
    image = Image.fromarray(image)  # Convert numpy array to PIL Image
    image = ImageOps.grayscale(image)  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors (if needed)
    image_array = image.resize((28, 28))  # Reshape to match model input
    image_array = np.array(image_array)  # Convert to numpy array
    image_array = image_array / 255.0  # Normalize pixel values
    image_array = np.array(image).reshape(1, 28, 28, 1)  # Add batch and channel dimensions

    #make prediction
    prediction = loaded_model.predict(image_array)
    predicted_digit = np.argmax(prediction)  # Get the predicted digit
    return int(predicted_digit)

# Define the Gradio interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(scale=2,),  # Users can draw digits here

    outputs="text",  # Output will be the predicted digit
    live=True,
    title="Digit Recognition App",
    description="Draw a digit (0–9) and get an instant prediction!"
)

if __name__ == "__main__":
    interface.launch(share=True, server_name="127.0.0.1", server_port=7860)  # Set share=True to allow public access
