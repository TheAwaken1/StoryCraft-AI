import gradio as gr
from datetime import datetime 
from pydub import AudioSegment 
from pydub.exceptions import CouldntDecodeError
import nltk 
import gc
import torch
from TTS.api import TTS
import time
import os
import numpy as np
from scipy.io.wavfile import write as write_wav
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
import io
import re
from huggingface_hub import snapshot_download
import warnings
import ssl

# Suppress deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Disable SSL verification for NLTK download (if needed for network issues)
try:
    _create_unverified_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_context

# Set NLTK data path to app/nltk_data
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
nltk.data.path.append(NLTK_DATA_PATH)

def ensure_punkt_data(max_retries=3, delay=5):
    """Ensure NLTK punkt data is available, with retries."""
    for attempt in range(max_retries):
        try:
            print("Checking for NLTK 'punkt' data...")
            nltk.data.find('tokenizers/punkt')
            print("'punkt' data found.")
            return True
        except LookupError:
            print(f"NLTK 'punkt' data not found. Attempting download (attempt {attempt + 1}/{max_retries})...")
            try:
                nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=True)
                nltk.data.find('tokenizers/punkt')
                print("'punkt' data downloaded successfully.")
                return True
            except Exception as e:
                print(f"Error downloading NLTK 'punkt': {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
    print("ERROR: Failed to find or download NLTK 'punkt' data after maximum retries.")
    raise RuntimeError("Could not load NLTK 'punkt' data. Please check your network connection or manually download 'punkt' to the nltk_data directory.")

# Ensure punkt data is available at startup
ensure_punkt_data()

# --- Configuration ---
# --- Model Definitions ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
print(f"BASE_DIR: {BASE_DIR}")
print(f"MODEL_DIR: {MODEL_DIR}")
AVAILABLE_MODELS = {
    "Phi-3 Mini (Fast, ~4GB VRAM)": {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct",
        "local_path": os.path.join(MODEL_DIR, "Phi-3-mini-4k-instruct"),
        "context_length": 4096,
        "quantized": False
    },
    "Qwen 2 7B Instruct (Balanced, ~6GB VRAM)": {
        "repo_id": "Qwen/Qwen2-7B-Instruct",
        "local_path": os.path.join(MODEL_DIR, "Qwen2-7B-Instruct"),
        "context_length": 8192,
        "quantized": False
    }
}
model_choices = list(AVAILABLE_MODELS.keys())
DEFAULT_MODEL_NAME = model_choices[0]

# Supported languages for text and TTS
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Korean": "ko"
}
language_choices = list(SUPPORTED_LANGUAGES.keys())
DEFAULT_LANGUAGE = "English"

# --- Global State for Loaded Models ---
current_llm_model = None
current_tokenizer = None
current_llm_name = None
tts_model = None

# --- Global Settings ---
USE_CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA_AVAILABLE else "cpu"

def download_model(repo_id, local_path):
    """Downloads model from Hugging Face Hub if not present locally."""
    if not os.path.exists(local_path):
        print(f"Downloading model {repo_id} to {local_path}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                cache_dir=os.path.join(BASE_DIR, "cache")
            )
            print(f"Model {repo_id} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model {repo_id}: {e}")
            raise RuntimeError(f"Failed to download model {repo_id}: {e}")
    else:
        print(f"Model {repo_id} already exists at {local_path}.")

def unload_llm_model():
    """Unloads the current LLM model and clears GPU cache."""
    global current_llm_model, current_tokenizer, current_llm_name
    if current_llm_model is not None:
        llm_name_unloading = current_llm_name
        print(f"Unloading previous LLM: {llm_name_unloading}")
        del current_llm_model
        del current_tokenizer
        current_llm_model = None
        current_tokenizer = None
        current_llm_name = None
        gc.collect()
        if USE_CUDA_AVAILABLE:
            torch.cuda.empty_cache()
            print("Cleared PyTorch CUDA cache.")
    else:
        print("No LLM model currently loaded to unload.")

def get_llm(model_name_to_load):
    """Loads the selected LLM model, downloading if necessary, unloading previous if different."""
    global current_llm_model, current_tokenizer, current_llm_name

    if model_name_to_load == current_llm_name and current_llm_model is not None:
        print(f"LLM '{model_name_to_load}' is already loaded.")
        return current_llm_model, current_tokenizer

    if current_llm_name is not None and current_llm_name != model_name_to_load:
        unload_llm_model()

    if model_name_to_load not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model selected: {model_name_to_load}")

    config = AVAILABLE_MODELS[model_name_to_load]
    repo_id = config["repo_id"]
    local_path = config["local_path"]
    is_quantized = config["quantized"]

    # Download model if not present
    download_model(repo_id, local_path)

    print(f"Loading LLM model: {model_name_to_load} from: {local_path}")
    try:
        # Configure 4-bit quantization for all models
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        # Set pad_token to avoid attention mask warning
        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

        current_llm_model = model
        current_tokenizer = tokenizer
        current_llm_name = model_name_to_load
        print(f"LLM model '{model_name_to_load}' loaded successfully.")
        return current_llm_model, current_tokenizer
    except Exception as e:
        print(f"Error loading LLM model '{model_name_to_load}': {e}")
        current_llm_model = None
        current_tokenizer = None
        current_llm_name = None
        raise RuntimeError(f"Failed to load LLM model '{model_name_to_load}': {e}")

# --- TTS Configuration ---
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
USE_CUDA_TTS = torch.cuda.is_available()

# Global variable for TTS model (lazy loading)
tts_model = None

def get_tts_model():
    """Loads the Coqui TTS model, only loads once."""
    global tts_model
    if tts_model is None:
        print(f"Loading TTS model: {TTS_MODEL_NAME} (GPU: {USE_CUDA_TTS})")
        print("This might take a while on first run to download the model...")
        os.environ["COQUI_TTS_AGREE_LICENSE"] = "1"
        print(f"COQUI_TTS_AGREE_LICENSE set to: {os.environ.get('COQUI_TTS_AGREE_LICENSE')}")
        try:
            sys.stdin = io.StringIO("y\n")
            tts = TTS(TTS_MODEL_NAME)
            sys.stdin = sys.__stdin__
            if USE_CUDA_TTS:
                tts.to("cuda")
            else:
                tts.to("cpu")
            tts_model = tts
            print("TTS model loaded successfully.")
            if hasattr(tts_model, 'speakers') and tts_model.speakers:
                print("Available built-in speakers:", tts_model.speakers)
            elif hasattr(tts_model, 'synthesizer') and hasattr(tts_model.synthesizer, 'speaker_manager') and hasattr(tts_model.synthesizer.speaker_manager, 'speaker_names'):
                print("Available built-in speakers:", tts_model.synthesizer.speaker_manager.speaker_names)
            else:
                print("Could not automatically list built-in speakers.")
        except Exception as e:
            print(f"Error loading TTS model: {e}")
            raise RuntimeError(f"Failed to load TTS model: {e}")
    return tts_model

def generate_story_llm(prompt, model_name, target_word_count, language, max_tokens=1800):
    """Generates story text in the specified language using the selected LLM."""
    story_text = f"Error: Failed during story generation with {model_name}."
    try:
        model, tokenizer = get_llm(model_name)
        config = AVAILABLE_MODELS[model_name]
        context_length = config["context_length"]

        # Customize system prompt for the selected language
        system_prompt = (
            f"You are a creative assistant who writes engaging short stories based on user prompts. "
            f"Write a story approximately {target_word_count} words long in {language}, with a rich narrative, vivid descriptions, "
            f"and a complete arc. Ensure the story concludes naturally within the word limit and is written entirely in {language}."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a short story based on this prompt: {prompt}"}
        ]

        # Convert messages to prompt with attention mask
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_ids = inputs.to(DEVICE)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(DEVICE)

        print(f"Generating story with {model_name} in {language} (Target: ~{target_word_count} words, max_tokens={max_tokens})...")
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        raw_story_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the story content, stripping prefixes like "assistant"
        story_start = raw_story_text.find(prompt) + len(prompt)
        story_text = raw_story_text[story_start:].strip() if story_start > len(prompt) else raw_story_text
        # Strip common prefixes like "assistant" or tokens
        prefixes = [r"^\s*assistant\s*:?\s*", r"^\s*Assistant\s*:?\s*", r"^\s*<\|\w+\|>\s*"]
        for prefix in prefixes:
            story_text = re.sub(prefix, "", story_text, flags=re.IGNORECASE)

        if not story_text:
            print("Warning: LLM returned empty response.")
            story_text = f"Error: LLM returned an empty story in {language}."
            return story_text

        actual_words = len(story_text.split())
        print(f"Generated story: {actual_words} words, {len(story_text)} characters")

        print("Checking sentence ending...")
        match = list(re.finditer(r'[.?!][\"\'”]?\s*$', story_text))
        is_complete_sentence = len(match) > 0

        if not is_complete_sentence:
            print("Potential partial sentence at end. Truncating...")
            end_indices = [m.end() for m in re.finditer(r'[.?!][\"\'”]?\s+', story_text)]
            if end_indices:
                last_end_index = max(end_indices)
                story_text = story_text[:last_end_index].strip()
                print(f"Truncated at index {last_end_index}.")
            else:
                print("Warning: No sentence ending found for truncation. Using full text.")
        else:
            print("Ends with complete sentence.")

        print(f"Story generation in {language} complete.")
        return story_text

    except Exception as e:
        print(f"Unexpected error during story generation for {model_name} in {language}: {e}")
        import traceback
        traceback.print_exc()
        return f"Unexpected error generating story with {model_name} in {language}: {e}"

def generate_speech_tts(text, voice_choice, speaker_wav_path=None, language="en", temperature=0.65, speed=1.0):
    """Generates speech in the specified language using Coqui TTS XTTSv2 model."""
    default_sr = 24000
    error_audio = (default_sr, np.zeros(int(default_sr * 0.5), dtype=np.int16))

    try:
        tts = get_tts_model()

        speaker_args = {}

        if not text:
            print("Warning: Empty text passed to TTS.")
            return error_audio

        print(f"Generating speech for text in {language}: {text[:70]}...")
        print(f"Selected voice choice: {voice_choice}")

        if voice_choice == "Clone from Sample":
            if speaker_wav_path and os.path.exists(speaker_wav_path):
                print(f"Using voice cloning with sample: {speaker_wav_path}")
                speaker_args = {"speaker_wav": [speaker_wav_path]}
            else:
                print("ERROR: 'Clone from Sample' selected, but no valid voice sample provided.")
                return error_audio
        else:
            available_speakers = []
            if hasattr(tts, 'speakers') and tts.speakers:
                available_speakers = tts.speakers
            elif hasattr(tts_model, 'synthesizer') and hasattr(tts_model.synthesizer, 'speaker_manager'):
                available_speakers = tts_model.synthesizer.speaker_manager.speaker_names

            if voice_choice in available_speakers:
                print(f"Using built-in speaker: {voice_choice}")
                speaker_args = {"speaker": voice_choice}
            else:
                print(f"ERROR: Selected speaker '{voice_choice}' not found in available speakers: {available_speakers}")
                return error_audio

        if not speaker_args:
            print("ERROR: No speaker reference determined (internal error).")
            return error_audio

        print(f"Splitting text into sentences for {language}...")
        sentences = nltk.sent_tokenize(text)

        wav_chunks = []
        try:
            sample_rate = tts.synthesizer.output_sample_rate
            if sample_rate is None: sample_rate = 24000
        except AttributeError:
            sample_rate = 24000

        print(f"Synthesizing sentences at {sample_rate} Hz in {language}...")
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence: continue

            print(f"Synthesizing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
            try:
                chunk = tts.tts(
                    text=sentence,
                    language=SUPPORTED_LANGUAGES.get(language, "en"),  # Use language code
                    temperature=float(temperature),
                    speed=float(speed),
                    **speaker_args
                )
                if chunk:
                    chunk_np = np.array(chunk, dtype=np.float32)
                    wav_chunks.append(chunk_np)
                else:
                    print(f"Warning: TTS returned empty data for sentence {i+1}")
            except Exception as sentence_e:
                print(f"Error synthesizing sentence {i+1} in {language}: {sentence_e}")

        if not wav_chunks:
            print("No audio chunks were generated.")
            return error_audio

        print("Concatenating audio chunks...")
        full_wav_np = np.concatenate(wav_chunks)
        full_wav_np_int16 = np.int16(full_wav_np * np.iinfo(np.int16).max)

        print(f"Speech generated successfully in {language}.")
        return (sample_rate, full_wav_np_int16)

    except Exception as e:
        print(f"Error during TTS generation in {language}: {e}")
        import traceback
        traceback.print_exc()
        return error_audio

def mix_audio_with_background(speech_audio_tuple, background_sound_name, volume_reduction_db=-18):
    """Mixes speech audio with a background sound file using pydub, with looping/trimming."""
    if speech_audio_tuple is None:
        print("Error: Cannot mix None audio.")
        return None
    speech_sr, speech_data = speech_audio_tuple
    if speech_data is None or speech_data.size == 0:
        print("Error: Cannot mix audio with empty data.")
        return speech_audio_tuple

    background_file_path = BACKGROUND_FILES.get(background_sound_name)

    if background_file_path is None:
        print("No background sound selected ('None'). Returning original speech.")
        return speech_audio_tuple

    if not os.path.exists(background_file_path):
        print(f"Warning: Background file not found: {background_file_path}. Skipping mixing.")
        return speech_audio_tuple

    try:
        print(f"Mixing speech (SR: {speech_sr}) with background: {background_sound_name}")

        if speech_data.dtype == np.int16:
            sample_width = 2
        elif speech_data.dtype == np.int32:
            sample_width = 4
        elif speech_data.dtype == np.float32:
            print("Converting float32 speech to int16 for pydub.")
            speech_data = np.int16(speech_data * np.iinfo(np.int16).max)
            sample_width = 2
        elif speech_data.dtype == np.float64:
            print("Converting float64 speech to int16 for pydub.")
            speech_data = np.int16(speech_data * np.iinfo(np.int16).max)
            sample_width = 2
        else:
            print(f"Warning: Unexpected speech data type {speech_data.dtype}. Attempting conversion to int16.")
            try:
                speech_data = speech_data.astype(np.int16)
                sample_width = 2
            except Exception as conv_e:
                print(f"ERROR: Failed to convert speech data to int16: {conv_e}")
                return speech_audio_tuple

        speech_segment = AudioSegment(
            data=speech_data.tobytes(),
            sample_width=sample_width,
            frame_rate=speech_sr,
            channels=1
        )
        target_duration_ms = len(speech_segment)

        print(f"Loading background file: {background_file_path}")
        background = AudioSegment.from_file(background_file_path)
        print(f"Background loaded: {len(background) / 1000:.2f}s, {background.frame_rate}Hz, {background.channels}ch")

        if background.frame_rate != speech_sr:
            print(f"Resampling background from {background.frame_rate}Hz to {speech_sr}Hz")
            background = background.set_frame_rate(speech_sr)
        if background.channels != 1:
            print("Converting background to mono")
            background = background.set_channels(1)

        background = background + volume_reduction_db
        print(f"Background volume reduced by {abs(volume_reduction_db)} dB")

        background_duration_ms = len(background)
        if background_duration_ms < target_duration_ms:
            times_to_loop = int(np.ceil(target_duration_ms / background_duration_ms))
            background = background * times_to_loop
            print(f"Looping background {times_to_loop} times")
            background = background[:target_duration_ms]
        elif background_duration_ms > target_duration_ms:
            background = background[:target_duration_ms]
            print(f"Trimming background to speech duration: {target_duration_ms / 1000:.2f}s")
        else:
            print("Background duration matches speech duration.")

        mixed_segment = background.overlay(speech_segment, position=0)
        print("Overlay complete.")

        final_sr = mixed_segment.frame_rate
        mixed_data_int16 = np.array(mixed_segment.get_array_of_samples()).astype(np.int16)

        print("Audio mixing complete.")
        return (final_sr, mixed_data_int16)

    except CouldntDecodeError:
        print(f"ERROR: Could not decode background file: {background_file_path}. Make sure ffmpeg is installed correctly and in PATH, and the audio file is valid.")
        return speech_audio_tuple
    except Exception as e:
        print(f"Error during audio mixing: {e}")
        import traceback
        traceback.print_exc()
        return speech_audio_tuple

def generate_story_audio(prompt, model_name_selection, language_selection, voice_choice, voice_sample_path, background_sound, target_minutes, temperature, speed):
    """Generates a story and converts it to audio in the specified language with optional background music."""
    status_text = ""
    story_text = "Error: Story generation did not complete."
    default_sr = 24000
    audio_output_value = (default_sr, np.zeros(int(default_sr * 0.1), dtype=np.int16))
    download_file_path = None
    output_filename = "generated_story.wav"
    output_dir = "output"
    output_filepath = os.path.join(output_dir, output_filename)

    try:
        WPM = 150
        target_word_count = int(target_minutes * WPM)
        selected_model_context_length = AVAILABLE_MODELS.get(model_name_selection, {}).get("context_length", 4096)
        context_buffer = 512
        max_allowed_tokens = max(selected_model_context_length - context_buffer, 512)
        max_tokens_for_llm = min(1800, max_allowed_tokens)

        status_text += f"Selected LLM: {model_name_selection}\n"
        status_text += f"Selected Language: {language_selection}\n"
        status_text += f"Generating story (Target: ~{target_minutes:.1f} min / ~{target_word_count} words, max_tokens={max_tokens_for_llm})...\n"
        story_text = "Generating..."
        yield status_text, story_text, None, None

        generated_story_result = generate_story_llm(
            prompt,
            model_name_selection,
            target_word_count,
            language_selection,
            max_tokens=max_tokens_for_llm
        )

        if not generated_story_result or not isinstance(generated_story_result, str) or generated_story_result.startswith("Error:"):
            error_msg = generated_story_result if isinstance(generated_story_result, str) else f"Error: LLM returned invalid result (Type: {type(generated_story_result)})."
            print(error_msg)
            status_text += error_msg + "\n"
            story_text = error_msg
            yield status_text, story_text, None, None
            return

        story_text = generated_story_result
        actual_words = len(story_text.split())
        status_text += f"Story generated ({actual_words} words).\n"
        yield status_text, story_text, None, None

        status_text += "Generating speech...\n"
        print(f"TTS input text: {story_text[:100]}")
        yield status_text, story_text, None, None
        actual_voice_sample_path = voice_sample_path if voice_sample_path else None
        generated_speech_audio = generate_speech_tts(
            story_text, voice_choice, actual_voice_sample_path, language=language_selection,
            temperature=temperature, speed=speed
        )
        audio_output_value = generated_speech_audio

        actual_duration_seconds = 0
        if audio_output_value and audio_output_value[1] is not None and audio_output_value[1].size > 0:
            sample_rate, audio_data = audio_output_value
            actual_duration_seconds = len(audio_data) / sample_rate
            status_text += f"Speech generated ({actual_duration_seconds:.1f} seconds).\n"
        else:
            status_text += "Speech generation failed or produced empty audio.\n"
        yield status_text, story_text, audio_output_value, None

        status_text += "Mixing audio...\n"
        yield status_text, story_text, audio_output_value, None
        final_audio_tuple = mix_audio_with_background(audio_output_value, background_sound)
        audio_output_value = final_audio_tuple
        status_text += "Audio mixing complete.\n"
        yield status_text, story_text, audio_output_value, None

        download_file_path = None
        if audio_output_value:
            sample_rate, audio_data = audio_output_value
            if audio_data is not None and audio_data.size > 0:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                output_filename = f"story_{timestamp}.wav"
                output_filepath = os.path.join(output_dir, output_filename)

                print(f"Saving final audio to: {output_filepath}")
                try:
                    write_wav(output_filepath, sample_rate, audio_data)
                    status_text += f"Final audio saved ({actual_words} words, {actual_duration_seconds:.1f}s): {output_filepath}\n"
                    download_file_path = output_filepath
                except Exception as write_e:
                    status_text += f"ERROR: Failed to write WAV file: {write_e}\n"
                    print(f"ERROR writing WAV: {write_e}")
                    download_file_path = None
            else:
                status_text += "Warning: Final audio data is empty or invalid. Skipping save.\n"
        else:
            status_text += "Warning: Audio generation failed or skipped. No file saved.\n"

        yield status_text, story_text, download_file_path, download_file_path

    except Exception as e:
        final_story_text_for_error_yield = story_text if 'story_text' in locals() and isinstance(story_text, str) else "Error: Processing failed."
        yield status_text, final_story_text_for_error_yield, None, None

# --- Define Voices, Backgrounds, and Model Selection ---
model_choices = list(AVAILABLE_MODELS.keys())
DEFAULT_MODEL_NAME = model_choices[0]
voice_options = ["Clone from Sample", 'Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski']
BACKGROUND_FILES = {
    "None": None,
    "Motivational": os.path.join(BASE_DIR, "backgrounds", "motivational_theme.mp3"),
    "Excitement": os.path.join(BASE_DIR, "backgrounds", "excitement_theme.mp3"),
}
background_sound_options = list(BACKGROUND_FILES.keys())

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎙️ StoryCraft AI")
    gr.Markdown("Created by [TheAwakenOne](https://x.com/TheAwakenOne619)")
    gr.Markdown(
        "This app was created as a hobby project. "
        "By using this app, you agree to the [Coqui Public Model License (CPML)](https://coqui.ai/cpml)."
    )
    gr.Markdown("Enter a prompt to generate a story in your chosen language, select a voice and background sound, and generate audio narration!")

    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(lines=5, label="Story Prompt", placeholder="e.g., A brave knight searching for a lost dragon...")
            model_select_input = gr.Dropdown(label="Choose LLM Model", choices=model_choices, value=DEFAULT_MODEL_NAME)
            language_select_input = gr.Dropdown(label="Choose Language", choices=language_choices, value=DEFAULT_LANGUAGE)
            voice_choice_input = gr.Dropdown(label="Voice Choice", choices=voice_options, value="Clone from Sample")
            voice_sample_input = gr.Audio(label="Voice Sample for Cloning (Optional WAV/MP3)", type="filepath")
            background_sound_input = gr.Dropdown(label="Background Sound", choices=background_sound_options, value="None")
            gr.Markdown("### TTS Tuning (Optional)")
            duration_slider = gr.Slider(minimum=1.0, maximum=5.0, step=0.5, value=3.0, label="Approx. Target Audio Duration (Minutes)")
            temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.65, label="Temperature (Higher=More Random)")
            speed_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Speed (1.0 = Normal)")
            generate_button = gr.Button("Generate Story & Audio", variant="primary")
        with gr.Column(scale=3):
            status_textbox = gr.Textbox(label="Status", interactive=False, lines=5)
            story_output_textbox = gr.Textbox(label="Generated Story", interactive=False, lines=8)
            audio_output = gr.Audio(label="Generated Audio Output", type="filepath")
            download_output = gr.File(label="Download Generated Audio")

    generate_button.click(
        fn=generate_story_audio,
        inputs=[
            prompt_input,
            model_select_input,
            language_select_input,
            voice_choice_input,
            voice_sample_input,
            background_sound_input,
            duration_slider,
            temperature_slider,
            speed_slider
        ],
        outputs=[status_textbox, story_output_textbox, audio_output, download_output]
    )
    app.unload(unload_llm_model)

if __name__ == "__main__":
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    app.launch(debug=True)