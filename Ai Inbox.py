import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import tempfile
import os


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

app_page = st.sidebar.selectbox("Select Page",["Home","About Project","Try AI inbox"])


if(app_page == "Home"):
    st.header('''Welcome to AI Inbox, A Music Genre Classification system!''')
    
    


# AI INBOX
elif(app_page == "Try AI inbox"):
    st.header("Model Prediction")
    mp3_file = st.file_uploader("Upload your Mp3 file here and AI Inbox will try to guess its genre", type=["mp3"])
    
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
                st.markdown(f"Model Prediction: **{label[result_index]}** music or song!")
            except Exception as e:
                st.error(f"Error occurred: {e}")