import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('model.hd5')
image_size = (64, 64)

def preprocess_image(image):
    # Convert the image to the required size and format
    img = image.resize(image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def main():
    st.title("Diabetic Retinopathy Detection App")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for prediction
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        probability = prediction[0][0]

        # Display the prediction result
        st.subheader("Prediction Result:")
        if probability > 0.4:
            st.success("Diabetic Retinopathy Detected!")
        else:
            st.success("No Diabetic Retinopathy Detected!")

        st.subheader("Probability:")
        st.write(f"{round(probability * 100, 2)}%")

if __name__ == "__main__":
    main()
