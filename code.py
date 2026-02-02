import tensorflow as tf
import tensorflow_hub as hub
import PIL
import numpy as np
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from googletrans import Translator

model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'

labels = r'C:\pythonProject1\landmarks_classifier_asia_V1_label_map.csv'
df = pd.read_csv(labels)

labels = dict(zip(df.id, df.name))
translator = Translator()


def image_processing(image):
    img_shape = (321, 321)
    classifier = hub.KerasLayer(model_url, input_shape=img_shape + (3,), output_key="predictions:logits")

    img = PIL.Image.open(image)
    img = img.resize(img_shape)
    img1 = img
    img = np.array(img) / 255.0
    img = img[np.newaxis]

    result = classifier(img)

    return labels[np.argmax(result)], img1


def get_map(loc):
    geolocator = Nominatim(user_agent="Your_Name")
    location = geolocator.geocode(loc)
    return location.address, location.latitude, location.longitude


def translate_address(address, target_lang):
    return translator.translate(address, dest=target_lang).text


def run():
    st.title("Landmark Recognition")
    img = PIL.Image.open(r'C:\pythonProject1\logo0.jpg')
    img = img.resize((256, 256))
    st.image(img)
    img_file = st.file_uploader("Choose your Image", type=['png', 'jpg'])

    if img_file is not None:
        save_image_path = './Uploaded_Images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        prediction, image = image_processing(save_image_path)
        st.image(image)
        st.header("📍 **Predicted Landmark is: " + prediction + '**')

        try:
            address, latitude, longitude = get_map(prediction)
            st.success('Address: ' + address)

            loc_dict = {'Latitude': latitude, 'Longitude': longitude}
            st.subheader('✅ **Latitude & Longitude of ' + prediction + '**')
            st.json(loc_dict)

            data = [[latitude, longitude]]
            df = pd.DataFrame(data, columns=['lat', 'lon'])
            st.subheader('✅ **' + prediction + ' on the Map**' + '🗺️')
            st.map(df)

            # Translation Feature
            st.subheader("🌍 Translate Address")
            languages = {"English": "en", "Spanish": "es", "French": "fr", "German": "de", "Chinese": "zh-cn",
                         "Arabic": "ar", "Hindi": "hi"}
            target_lang = st.selectbox("Select Language", list(languages.keys()))
            if address:
                translated_address = translate_address(address, languages[target_lang])
                st.success("Translated Address: " + translated_address)
        except Exception as e:
            st.warning("No address found!!")


run()
