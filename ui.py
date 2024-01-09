import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
from tensorflow.keras.models import load_model
import seaborn as sns

# Load the pre-trained model
model = load_model('plant_species_model.h5')  # Update with the correct path

# Constants
img_size = 224
flowers = ['banana', 'coconut', 'corn', 'mango', 'orange', 'paddy', 'papaya', 'pineapple', 'sweet potatoes', 'watermelon']

# Streamlit App
st.title("Plant Species Prediction")

# Upload Image through Streamlit
uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Read and preprocess the image
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_arr = cv2.resize(img, (img_size, img_size))
    st.image(resized_arr, caption="Uploaded Image.", use_column_width=True)

    # Model Prediction
    x = preprocess_input(np.array([resized_arr]))
    x = x.reshape(-1, img_size, img_size, 3)
    pred = model.predict(x)
    prediction_label = flowers[np.argmax(pred)]

    # Display prediction result
    st.subheader("Prediction:")
    if prediction_label == 'banana':
        st.write("The predicted plant species is: banana")
        st.write("Bananas are fruiting plants that grows near tropical rainforests. There are many banana species, but Musa acuminata varieties are most common in the US. Bananas contain fiber, potassium, vitamin B6, and carbohydrates. Ripe bananas contain more soluble fiber, while unripe bananas contain more insoluble fiber. All parts of the banana plant, including the fruit, leaves, stem, flowers, and roots, have been used as medicine. People use bananas for diarrhea. They are also used for athletic performance, constipation, diabetes, high cholesterol, obesity, and many other conditions, but there is no good scientific evidence to support most of these uses.")
        st.subheader("Usage:")
        st.write("Diarrhea. Eating cooked green bananas might help reduce diarrhea symptoms in young children.")
        st.subheader("Side Effects:")
        st.write("When taken by mouth: Bananas are commonly eaten as food. They're generally well-tolerated, but some people might experience bloating, gas, or cramping. There isn't enough reliable information to know if other parts of the banana plant are safe or what the side effects might be.")
        st.write("When applied to the skin: Banana leaves are possibly safe when used short-term. Some people are allergic to banana and might develop a rash or hives. There isn't enough reliable information to know if other parts of the banana plant are safe or what the side effects might be.")
    elif prediction_label == 'coconut':
        st.write("The predicted plant species is: coconut")
        st.write("Coconut (Cocos nucifera) is the fruit of the coconut palm, which grows in tropical places around the world. It can be eaten as food or used as medicine. Coconuts contain a high amount of a saturated fat called medium chain triglycerides. These fats work differently than other types of saturated fat in the body. They might increase fat burning and reduce fat storage. Coconut flour, which is made from coconut, contains high amounts of dietary fiber. People use coconut for diabetes, high cholesterol, obesity, and other conditions, but there is no good scientific evidence to support these uses.")
        st.subheader("Usage:")
        st.write("Diabetes. Early research suggests that taking coconut oil with meals slightly reduces blood sugar levels compared to eating alone in people with and without diabetes. But it's unclear if this small reduction is enough to have a measurable effect. Also, it's unclear if coconut oil is beneficial for people with diabetes.")
        st.write("High cholesterol. Early research suggests that eating coconut oil for 8 weeks might reduce total cholesterol, low-density lipoprotein (LDL or \"bad\") cholesterol, and triglycerides in people with coronary artery disease. But coconut oil does not seem to increase high-density lipoprotein (HDL or \"good\") cholesterol. Also, eating coconut meat along with a low-calorie diet does not seem to reduce body weight, body mass index (BMI), or waist circumference compared to eating low-calorie alone in people with obesity.")
        st.write("Obesity. Early research suggests that eating coconut meat along with a low-calorie diet does not reduce body weight, body mass index (BMI), or waist circumference compared to eating low-calorie alone in people with obesity.")
        st.subheader("Side Effects:")
        st.write("When taken by mouth: Coconut is commonly consumed as food. Coconut is possibly safe when used as medicine, short-term. In some people, eating coconuts might cause an allergic reaction. Symptoms might include skin rashes and difficulty breathing.")
    elif prediction_label == 'corn':
        st.write("The predicted plant species is: corn")
        st.write("Corn is a starchy vegetable and cereal grain that has been eaten all over the world for centuries. It’s rich in fiber, vitamins and minerals. However, the health benefits of corn are controversial — while it contains beneficial nutrients, it can also spike blood sugar levels. This article tells you everything you need to know about corn.")
        st.subheader("Usage:")
        st.write("Corn is LIKELY SAFE for most people when taken by mouth in amounts commonly found in food. Corn is POSSIBLY SAFE when used in medicinal amounts (larger amounts than what is normally found in food). Special Precautions & Warnings: Pregnancy and breast-feeding: Corn is LIKELY SAFE when eaten in food amounts. There isn't enough reliable information to know if corn is safe to use in larger medicinal amounts when pregnant or breast-feeding. Stay on the safe side and stick to food amounts.")
        st.write("Diabetes: Corn might increase blood sugar levels in people with type 2 diabetes. Monitor your blood sugar closely if you have diabetes and use corn in amounts larger than the amounts you normally eat.")
        st.subheader("Side Effects:")
        st.write("Corn is a starchy vegetable, like potatoes and peas. That means it has sugar and carbohydrates that can raise your blood sugar levels. It can still be a healthy part of your diet if you don't overdo it. If you have diabetes, you don't necessarily need to avoid corn, but watch your portion sizes.")
    elif prediction_label == 'mango':
        st.write("The predicted plant species is: mango")
        st.write("Mangoes are a tropical fruit that contain nutrients that may help support a healthy heart, gut, eye, and hair health. Mangoes are a tropical fruit that are high in nutrients and antioxidants. They have been linked to many health benefits, including improved digestion, skin and hair health, lower cholesterol levels, and increased immunity.")
        st.subheader("Usage:")
        st.write("Mango is LIKELY SAFE for most people when eaten in food amounts. Mango is POSSIBLY SAFE when taken as a medicine by mouth in amounts up to 1 gram daily. Some people might experience stomach upset, vomiting, diarrhea, and allergic reactions.")
        st.write("Mango is POSSIBLY UNSAFE when applied directly to the skin. It might cause irritation, especially in sensitive people.")
        st.subheader("Side Effects:")
        st.write("After consuming certain mango species, some people may experience throat pain or allergy (stomach pain, sneezing and runny nose). Consuming mangoes in excess can result in gastrointestinal problems like stomach pain, indigestion and diarrhoea.")
    elif prediction_label == 'orange':
        st.write("The predicted plant species is: orange")
        st.write("Oranges are a type of low calorie, highly nutritious citrus fruit. As part of a healthful and varied diet, oranges contribute to strong, clear skin and can help lower a person’s risk of many conditions.")
        st.subheader("Usage:")
        st.write("Orange is LIKELY SAFE for adults, when taken by mouth in amounts commonly found in food. Orange is POSSIBLY SAFE when taken by mouth in medicinal amounts. In some people, orange can cause allergic reactions.")
        st.subheader("Side Effects:")
        st.write("Watch out for nausea, vomiting, diarrhea, stomach cramps, headache, and insomnia. Oranges are high in acid, and that can make symptoms of gastroesophageal reflux disease (GERD) worse. If you're taking beta-blockers, too many oranges could increase your potassium intake and lead to kidney damage.")
    elif prediction_label == 'paddy':
        st.write("The predicted plant species is: paddy")
        st.write("Rice is the seed of the grass species Oryza sativa (Asian rice) or less commonly Oryza glaberrima (African rice). The name wild rice is usually used for species of the genera Zizania and Porteresia, both wild and domesticated, although the term may also be used for primitive or uncultivated varieties of Oryza.")
        st.subheader("Usage:")
        st.write("Rice is LIKELY SAFE for most people when eaten in food amounts. Rice is POSSIBLY SAFE when taken by mouth as a medicine. Rice bran is LIKELY SAFE for most people when taken by mouth appropriately. Rice bran and rice bran oil are POSSIBLY SAFE when taken by mouth in medicinal amounts.")
        st.subheader("Side Effects:")
        st.write("Rice is a good source of energy and protein, but not all grains are created equal. White rice is a refined, high-carb food that's had most of its fiber removed. A high intake of refined carbs has been linked to obesity and chronic disease. Brown Versus White Rice.")
    
    elif prediction_label == 'papaya':
        st.write("The predicted plant species is: papaya")
        st.write("The papaya, papaw, or pawpaw is the plant Carica papaya, one of the 22 accepted species in the genus Carica of the family Caricaceae. Its origin is in the tropics of the Americas, perhaps from southern Mexico and neighboring Central America.")
        st.subheader("Usage:")
        st.write("Papaya is LIKELY SAFE for most people when taken by mouth in amounts commonly found in foods. Papaya is POSSIBLY SAFE when taken by mouth in medicinal amounts. Papaya latex can be a severe irritant and vesicant on skin. Papaya juice and papaya seeds are unlikely to cause adverse effects when taken orally; however, papaya leaves at high doses may cause stomach irritation.")
        st.subheader("Side Effects:")
        st.write("Papaya is POSSIBLY UNSAFE when taken by mouth in large amounts or when applied to the skin as papaya latex. Taking large amounts of papaya by mouth could damage the esophagus, which is the food tube in the throat. Applying papaya latex to the skin can cause severe irritation and allergic reactions in some people.")
    elif prediction_label == 'pineapple':
        st.write("The predicted plant species is: pineapple")
        st.write("The pineapple is a tropical plant with an edible fruit and the most economically significant plant in the family Bromeliaceae. The pineapple is indigenous to South America, where it has been cultivated for many centuries.")
        st.subheader("Usage:")
        st.write("Pineapple is LIKELY SAFE when consumed in amounts commonly found in foods. Pineapple is POSSIBLY SAFE when taken by mouth in medicinal amounts. However, it is not known if pineapple is safe when used in larger amounts as medicine. Some people are allergic to pineapple.")
        st.subheader("Side Effects:")
        st.write("Consuming too much pineapple can cause tenderness of the mouth as the fruit is a great meat tenderizer. Eating too much pineapples may cause tenderness of the mouth as the fruit is a great meat tenderizer. Eating too much pineapples may cause tenderness of the mouth.")
    elif prediction_label == 'sweet potatoes':
        st.write("The predicted plant species is: sweet potatoes")
        st.write("The sweet potato or sweetpotato is a dicotyledonous plant that belongs to the bindweed or morning glory family, Convolvulaceae. Its large, starchy, sweet-tasting, tuberous roots are a root vegetable. The young leaves and shoots are sometimes eaten as greens.")
        st.subheader("Usage:")
        st.write("Sweet potato is LIKELY SAFE for most people when taken by mouth in amounts commonly found in food. Sweet potato is POSSIBLY SAFE when used as a medicine, short-term. The safety of using sweet potato extract long-term is unknown.")
        st.subheader("Side Effects:")
        st.write("Sweet potato contains significant amounts of oxalates that can cause kidney stones. Continuous eating of sweet potatoes can cause kidney stones. Sweet potato contains oxalates that can increase the amount of oxalate in the urine. Oxalates can increase the risk of kidney stones.")
    elif prediction_label == 'watermelon':
        st.write("The predicted plant species is: watermelon")
        st.write("Watermelon is a sweet and refreshing low calorie summer snack. It provides hydration and also essential nutrients, including vitamins, minerals, and antioxidants.")
        st.subheader("Usage:")
        st.write("Watermelon is LIKELY SAFE when taken by mouth in amounts commonly found in food. Watermelon is POSSIBLY SAFE when taken by mouth as a medicine for up to 6 weeks. Watermelon contains a lot of water. Diuretics such as chlorothiazide (Diuril) and hydrochlorothiazide (HydroDIURIL, Esidrix) increase how much urine the body makes and can lead to dehydration. Taking watermelon along with diuretics might decrease potassium in the body too much.")
        st.subheader("Side Effects:")
        st.write("Eating large amounts of watermelon may have a laxative effect in some people. It contains a lot of water and a small amount of fiber — both of which are important for healthy digestion. Eating too much watermelon may cause your body to lose water, which is especially dangerous for people with diabetes.")
    else:
        pass
    # Display prediction probabilities
    pred_results = pd.DataFrame(data=pred, columns=flowers)
    st.subheader("Prediction Probabilities:")
    st.bar_chart(pred_results.T)

# Note: Adjust the model loading part based on the actual method used to load your model.
