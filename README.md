# Image Search with ChromaDB and Gemini

This project is a web application that allows users to search for images using text or audio queries. The application uses ChromaDB for image indexing and retrieval, and the Gemini API for generating text embeddings. Users can input their queries via text or microphone, and the application will return the most relevant images along with audio descriptions.

## Features

- **Text Query**: Users can input a text query to search for images.
- **Audio Query**: Users can use their microphone to input an audio query, which will be transcribed to text and used for the search.
- **Image Descriptions**: The application generates audio descriptions for each image result.
- **Real-time Audio Streaming**: Uses `streamlit-webrtc` for real-time audio streaming from the microphone.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    Create a `.env` file in the root directory and add the following:
    ```env
    API_KEY=your_gemini_api_key
    ```

## Usage

### Creating a Vector Database

To create a vector database from a dataset of images, use the `create_vectordb.py` script. This script takes two arguments: the path to the dataset folder and the name of the vector database.

1. **Run the script**:
    ```sh
    python create_vectordb.py /path/to/dataset_folder my_vectordb_name
    ```

    Replace `/path/to/dataset_folder` with the actual path to your dataset folder and `my_vectordb_name` with the desired name for your ChromaDB vector database.

### Running the Streamlit Application

1. **Run the Streamlit application**:
    ```sh
    streamlit run streamlit_app.py
    ```

2. **Open your web browser** and go to `http://localhost:8501`.

3. **Enter your query**:
    - **Text Query**: Enter your query text in the input box.
    - **Audio Query**: Click on the microphone button to start recording your query.

4. **View Results**: The application will display the most relevant images along with audio descriptions.

## File Structure
project/ │ ├── app/ │ ├── init.py │ ├── config.py │ ├── routes.py │ ├── utils.py │ └── templates/ │ └── index.html │ ├── static/ │ └── images/ │ ├── run.py ├── streamlit_app.py ├── requirements.txt └──


## Dependencies

- `streamlit`
- `torch`
- `torchvision`
- `Pillow`
- `numpy`
- `google-generativeai`
- `chromadb`
- `llama-index`
- `SpeechRecognition`
- `gtts`
- `streamlit-webrtc`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Google Generative AI](https://cloud.google.com/ai-platform/generative-ai)
- [ChromaDB](https://chromadb.com/)
- [Google Text-to-Speech](https://pypi.org/project/gTTS/)
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)

