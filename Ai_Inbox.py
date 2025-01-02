import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import tempfile
import os
from pathlib import Path


#load model function
st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("./training_hist_and_models/Training_model.h5")
    return model

def load_and_preprocess_file(file_path,target_shape=(150,150)):
    data= []
    audio_data,sample_rate = librosa.load(file_path,sr=None)
    chunk_duration = 4
    overlap_duration = 2
    #convert duration to sample
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    #calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    #iterate each chunk
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        with tf.device('/CPU:0'):  # Resize on CPU
            mel_spectrogram = tf.image.resize(
                np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    return np.array(data)


##MODEL PREDICTION
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements,counts = np.unique(predicted_categories,return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts==max_count]
    return max_elements[0]


#UI
st.sidebar.title("Dashboard")

app_page = st.sidebar.selectbox("Select Page",["Home","About Project"])


if app_page == "Home":
    st.header("Welcome to AI Inbox, A Music Genre Classification system!")
    st.markdown("""
    **AI Inbox** is an innovative music genre classification system powered by artificial intelligence. 
    Upload your audio file, and let our model predict the genre of your song!

    ### How it works:
    1. Upload an audio file in .mp3 format.
    2. Click on the 'Play Audio' button to listen to your file.
    3. Click 'Predict Genre' to see the genre of the song.

    It's that simple! Try it now and let AI Inbox classify your song's genre!
    """)
    
    st.header("Model Prediction")
    mp3_file = st.file_uploader("Upload your Mp3 file here and AI Inbox will try to guess its genre(Note:genres are limited to 10 genres, see about us for the genre list available)", type=["mp3"])
    
    if mp3_file is not None:
        # Save the file to a temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(mp3_file.read())  # Save the uploaded file
            file_path = tmp_file.name  # Get the temporary file path

        st.markdown(f"File saved to temporary path!")  # Inform user

        # Optionally play the audio
        if st.button("Play Audio"):
            st.audio(file_path)
    
    if st.button("Predict Genre"):
        with st.spinner("Please wait Ai Inbox is analyzing..."):
            try:
                X_test = load_and_preprocess_file(file_path)  # Load and preprocess the saved file
                result_index = model_prediction(X_test)  # Predict the genre
                
                label = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
                st.balloons()
                st.markdown(f"**Model Prediction: :orange[{label[result_index]}] music!**")
            except Exception as e:
                st.error(f"Error occurred: {e}")
    


elif app_page == "About Project":
    group_members = [
    {
        "name": "Dexter C. Pacana",
        "role": "Video Documentation and Editing Lead",
        "image_path":"./Members/member1.webp",
    },
    {
        "name": "Renie Boy M. Maglinte",
        "role": "Documentation Specialist",
        "image_path":"./Members/member2.webp",
    },
    {
        "name": "Lyndon King L. Obenza",
        "role": "Developer",
        "image_path":"./Members/member3.webp",
    }
    
]
    st.header("About AI Inbox")
    st.markdown("""
    **AI Inbox** leverages machine learning techniques to classify songs into various genres. 
    This system was trained on a large dataset of music, and it uses Mel spectrogram features to analyze the audio and determine its genre. 
    The goal is to help music lovers and creators categorize and identify songs with ease.

    ### Technical Details:
    - The model is built using Convolutional Neural Networks (CNNs) and trained on a collection of Mel spectrograms from 10 music genres.
    -This  genres are blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock
    - We preprocess the audio files into Mel spectrograms and resize them to fit the model's input requirements.

    ### Libraries & Frameworks:
    - **TensorFlow**: Provides powerful deep learning capabilities for model building and prediction.
    - **Librosa**: Used for audio feature extraction and preprocessing.
    - **Streamlit**: for this interactive web application.
    
    ### Version Used:
    - **Python**:3.10.16
    - **tensorflow**:2.10.0
    - **scikit-learn**:1.3.0
    - **numpy**:1.24.3
    - **matplotlib**:3.7.2
    - **seaborn**:0.13.0
    - **pandas**:2.1.0
    - **streamlit**:1.22.0
    - **librosa**:0.10.1

    ### Credits:
    - This project was made possible by open-source dataset and libraries,the GTZAN dataset and the Librosa library.
    - Special thanks to the TensorFlow community for their valuable resources.
    - Youtube channels who became our guide for this project to be completed such as but not limited to IBM Technology,SPOTLESS TECH,James Briggs,Aladdin Persson,codebasics and others.
    
    """)
    
    
    st.markdown("""
                ***Document link***
                - https://docs.google.com/document/d/1HYeWml-7T1M3dThNWwtiPX-I3P84di4zBtdWK96FicA/edit?tab=t.0
                
                **Repository Link**
                - https://github.com/Leonowki/Music_Genre_Classification
                
                
                
                
                """)
    
    # Display member details
    for member in group_members:
        col1, col2 = st.columns([1, 3])  

        with col1:
            st.image(member["image_path"], caption=member["name"], use_column_width=True)
        
        with col2:
            st.subheader(member["name"])
            st.write(f"**Role:** {member['role']}")
        
    



    