import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('brain.h5')

# Class labels
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def load_and_predict(image):
    # Preprocess the image for prediction
    image = cv2.resize(image, (150, 150))  # Resize the image to match the input shape of the model
    image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch size

    # Make predictions
    predictions = model.predict(image)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_idx]

    return predicted_class

def main():
    st.title("Brain Tumor Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image.", width=200)

        if st.button("Predict"):
            predicted_class = load_and_predict(image)
            st.success(f"Predicted Class: {predicted_class}")
            st.divider()
            if predicted_class=='glioma_tumor':
                st.write('Glioma is a growth of cells that starts in the brain or spinal cord. The cells in a glioma look similar to healthy brain cells called glial cells. Glial cells surround nerve cells and help them function. As a glioma grows it forms a mass of cells called a tumor. The tumor can grow to press on brain or spinal cord tissue and cause symptoms. Symptoms depend on which part of the brain or spinal cord is affected.')
                st.write('Symptoms:')
                st.write("Headache, particularly one that hurts the most in the morning")
                st.write("Nausea and vomiting.")
                st.write("Confusion or a decline in brain function, such as problems with thinking and understanding information.")

            elif predicted_class=='meningioma_tumor':
                st.write("A meningioma is a tumor that grows from the membranes that surround the brain and spinal cord, called the meninges. A meningioma is not a brain tumor, but it may press on the nearby brain, nerves and vessels. Meningioma is the most common type of tumor that forms in the head. Most meningiomas grow very slowly. They can grow over many years without causing symptoms. But sometimes, their effects on nearby brain tissue, nerves or vessels may cause serious disability.")
                st.write('Symptoms:')
                st.write("Changes in vision, such as seeing double or blurring.")
                st.write("Headaches that are worse in the morning.")
                st.write("Hearing loss or ringing in the ears.")

            elif predicted_class=='no_tumor':
                st.write("No tumor detected.")
            elif predicted_class=='pituitary_tumor':
                st.write("Pituitary tumors are unusual growths that develop in the pituitary gland. This gland is an organ about the size of a pea. It's located behind the nose at the base of the brain. Some of these tumors cause the pituitary gland to make too much of certain hormones that control important body functions. Others can cause the pituitary gland to make too little of those hormones. Most pituitary tumors are benign. That means they are not cancer. Another name for these noncancerous tumors is pituitary adenomas. Most adenomas stay in the pituitary gland or in the tissue around it, and they grow slowly. They typically don't spread to other parts of the body.")
                st.write('Symptoms:')
                st.write("Headache.")
                st.write("Eye problems due to pressure on the optic nerve, especially loss of side vision, also called peripheral vision, and double vision.")
                st.write("Pain in the face, sometimes including sinus pain or ear pain.")




if __name__ == "__main__":
    main()
