import streamlit as st
import os
import logging
from typing import List, Tuple
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import json
import torch
import speech_recognition as sr
from gtts import gTTS
import tempfile
from keybert import KeyBERT

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
torch.classes.__path__ = []
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ChromaDB configurations
vectordb =  os.getenv("vectordb")


# Load the BLIP model and processor
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def get_image_description(image_path: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = blip_processor(images=image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        description = blip_processor.decode(out[0], skip_special_tokens=True)
        print(description)
        return description
    except Exception as e:
        logger.error(f"Error getting image description: {e}")
        raise

def generate_categories(query, top_n=5):
    """
    Generates categories based on a given query using KeyBERT for keyword extraction.
    
    :param query: str, input query
    :param top_n: int, number of categories to return
    :return: list of extracted keywords as category suggestions
    """
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    
    return [keyword[0] for keyword in keywords]

def search_images_chromadb(query_text, k: int, collection) -> List[Tuple[str, float]]:
    try:
        prompt = "The following is a query for an image in a database, find relevant categories for the prompt. Output must be a dictionary with key value pair 'categories' and list of categories as the value"

        # Use a text generation pipeline from Hugging Face
        #text_generator = pipeline("text-generation", model="gpt2")
        #response = text_generator(prompt + query_text, max_length=50)
        #print(response)
        #query_list = json.loads(response[0]['generated_text'].split('json')[1].replace("\n", "").replace("`", ""))
        query_list = generate_categories(query_text)
        print(query_list)
        results = collection.query(query_texts=[query_text],
                                   n_results=k,
                                   include=['documents', 'distances', 'metadatas', 'data', 'uris'])
        image_path = rank_search_results(results)
        print(image_path)
        return [path[0] for path in image_path[:k]]
    except Exception as e:
        logger.error(f"Error during image search: {e}")
        raise

def rank_search_results(results) -> List[Tuple[str, float]]:
    ranked_results = []
    print("Results:", results)  # Debug print to check the structure of results
    for result in results['metadatas']:
        print("Metadata result:", result)  # Debug print to check each metadata result
        for doc, dist in zip(result, results['distances'][results['metadatas'].index(result)]):
            print("Doc:", doc)  # Debug print to check each document
            print("Dist:", dist)  # Debug print to check each distance
            ranked_results.append((doc['path'], dist))
    ranked_results.sort(key=lambda x: x[1], reverse=False)
    print("Ranked results:", ranked_results)  # Debug print to check the final ranked results
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
    client = chromadb.PersistentClient(path=vectordb)
    image_loader = ImageLoader()
    multimodal_ef = OpenCLIPEmbeddingFunction()
    collection = client.get_or_create_collection(name=index_name, embedding_function=multimodal_ef, data_loader=image_loader)
    print("Collection loaded")
    # Search for nearest neighbors
    return search_images_chromadb(query_text, k, collection)

# Streamlit app
st.title("Image Search with ChromaDB and Open-Source Models")

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
            description = get_image_description(img_path)
            audio_file_path = generate_audio_description(description)
            st.audio(audio_file_path)
    except Exception as e:
        st.error(f"Error finding top k images: {e}")