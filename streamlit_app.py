import streamlit as st
import os
import logging
from typing import List, Tuple
import chromadb
from PIL import Image
import google.generativeai as genai
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import json
import torch
import speech_recognition as sr
from gtts import gTTS
import tempfile


torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
torch.classes.__path__ = []
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini API with Gemini Pro model
API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)


def get_image_description( image_path: str) -> str:
    prompt = "Explain what is going on in the image. Output must be a string."
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash"
        )
        image = Image.open(image_path)
        response = model.generate_content([prompt, image])
        print(response.text)
        return response.text#json.loads(response.text.split('json')[1].replace("\n","").replace("`",""))['text']
    except Exception as e:
        logger.error(f"Error getting image description: {e}")
        raise


def search_images_chromadb(query_text, k: int, collection) -> List[Tuple[str, float]]:
    try:
        prompt = "The following is a query for an image in a database, find relevant categories for the prompt. Output must be  a dictionary with key value pair 'categories' and list of categories as the value"

        model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash"        )
        response = model.generate_content([prompt, query_text])
        query_list = json.loads(response.text.split('json')[1].replace("\n","").replace("`",""))
        results = collection.query(query_texts = query_list['categories'],
    n_results=k,
    include=['documents', 'distances', 'metadatas', 'data', 'uris'])
        image_path = rank_search_results(results)
        return [path[0] for path in image_path[:k]]
    except Exception as e:
        logger.error(f"Error during image search: {e}")
        raise
def rank_search_results(results) -> List[Tuple[str, float]]:
    ranked_results = []
    for result in results['metadatas']:
        for doc, dist in zip(result, results['distances'][results['metadatas'].index(result)]):
            ranked_results.append((doc['path'], dist))
    ranked_results.sort(key=lambda x: x[1], reverse=False)
    return ranked_results

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
    return ""
def generate_audio_description(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

def find_top_k_images(query_text, index_name="multimodal_db", k=5):
    """Retrieve top K similar images from ChromaDB."""


    # Load ChromaDB collection
    client = chromadb.PersistentClient(path="my_vectordb_test")
    image_loader = ImageLoader()
    multimodal_ef = OpenCLIPEmbeddingFunction()
    collection = client.get_or_create_collection(name=index_name, embedding_function=multimodal_ef, data_loader=image_loader)

    # Search for nearest neighbors
    return search_images_chromadb(query_text, k, collection)

# Streamlit app
st.title("Image Search with ChromaDB and Gemini")

query_text = st.text_input("Enter your query text:")
audio_value = st.audio_input("Record a voice message")

if audio_value:
    query_text = transcribe_audio(audio_value)

k = st.slider("Number of results", 1, 10, 5)

if st.button("Search"):
    try:
        top_k_results = find_top_k_images(query_text, index_name="multimodal_db", k=k)
        for img_path in top_k_results:
            st.image(img_path)
            description= get_image_description(img_path)
            audio_file_path = generate_audio_description(description)
            st.audio(audio_file_path)
    except Exception as e:
        st.error(f"Error finding top k images: {e}")