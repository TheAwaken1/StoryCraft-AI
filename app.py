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
    "Qwen 2.5 7B Instruct (Balanced, ~6GB VRAM)": { # Updated to Qwen 2.5
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_path": os.path.join(MODEL_DIR, "Qwen2.5-7B-Instruct"),
        "context_length": 32768, # Max, can be practically limited in app
        "quantized": False
    },
    "Llama 3 8B Instruct (Meta, ~7GB VRAM)": { # Note: Gated, requires HF_TOKEN
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "local_path": os.path.join(MODEL_DIR, "Meta-Llama-3-8B-Instruct"),
        "context_length": 8192,
        "quantized": False
    },
    "Nous Hermes 2 Pro Llama 3 (Llama 3 Fine-tune, ~7GB VRAM)": { # Note: Gated via Llama 3 base
        "repo_id": "NousResearch/Hermes-2-Pro-Llama-3-8B",
        "local_path": os.path.join(MODEL_DIR, "Hermes-2-Pro-Llama-3-8B"),
        "context_length": 8192,
        "quantized": False
    },
    "Google Gemma 2 2B Instruct (Small, ~3GB VRAM)": { # Note: Gated, requires HF_TOKEN
        "repo_id": "google/gemma-2-2b-it",
        "local_path": os.path.join(MODEL_DIR, "gemma-2-2b-it"),
        "context_length": 8192,
        "quantized": False
    },
    # --- RELIABLE NON-GATED HIGH-PERFORMANCE OPTION ---
    "Qwen 2.5 14B Instruct (High Quality, Non-Gated, ~10GB VRAM)": {
        "repo_id": "Qwen/Qwen2.5-14B-Instruct",
        "local_path": os.path.join(MODEL_DIR, "Qwen2.5-14B-Instruct"),
        "context_length": 65536, # Max, can be practically limited
        "quantized": False
    }
}
model_choices = list(AVAILABLE_MODELS.keys())
# Set a reliable default, perhaps one of the non-gated ones if HF_TOKEN is an issue for some users
DEFAULT_MODEL_NAME = "Qwen 2.5 7B Instruct (Balanced, ~6GB VRAM)"
# Or if you want the new larger one as default:
# DEFAULT_MODEL_NAME = "Qwen 2.5 14B Instruct (High Quality, Non-Gated, ~10GB VRAM)"

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
            # snapshot_download will automatically use the HF_TOKEN environment variable
            # if it's set and the model is gated.
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                cache_dir=os.path.join(BASE_DIR, "cache")
                # You could explicitly pass the token here too, but environment variable is standard:
                # token=os.environ.get("HF_TOKEN")
            )
            print(f"Model {repo_id} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model {repo_id}: {e}")
            # More specific error handling for gated models might be useful here
            if "401 Client Error" in str(e) or "Gated model" in str(e):
                print(f"This might be a gated model. Ensure you have accepted the terms on Hugging Face and your HF_TOKEN is correctly set with the necessary permissions.")
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
        repo_id = config.get("repo_id", "").lower() # Get repo_id for checking model type

        # Customize system prompt for the selected language
        base_instruction = (
            f"You are a creative assistant who writes engaging short stories based on user prompts. "
            f"Write a story approximately {target_word_count} words long in {language}, with a rich narrative, vivid descriptions, "
            f"and a complete arc. Ensure the story concludes naturally within the word limit and is written entirely in {language}."
        )

        user_content = f"Write a short story based on this prompt: {prompt}"

        messages = []
        # Check if the model is Gemma and adjust message format accordingly
        # (You might want a more robust way to check model type if you add many more)
        if "gemma" in repo_id:
            # For Gemma, combine system-like instructions with the user prompt
            # Gemma's template usually doesn't support a separate 'system' role.
            # The prompt should be structured to guide the model directly.
            # The apply_chat_template for Gemma will expect 'user' and 'model' turns.
            # Prepending instructions to the user prompt is a common workaround.
            full_user_prompt = f"{base_instruction}\n\nUser prompt: {prompt}"
            messages = [
                {"role": "user", "content": full_user_prompt}
            ]
            print(f"Using Gemma-specific chat format (no system role). Combined prompt: {full_user_prompt[:200]}...")
        else:
            # Standard format for models that support system role (like Llama 3, Qwen, Phi-3)
            messages = [
                {"role": "system", "content": base_instruction},
                {"role": "user", "content": user_content}
            ]
            print(f"Using standard chat format with system role. System prompt: {base_instruction[:100]}...")


        # Convert messages to prompt with attention mask
        # Make sure the tokenizer has a chat_template, otherwise apply_chat_template might fail
        if tokenizer.chat_template is None:
            # Fallback for models without a chat_template (less ideal)
            # This part might need adjustment based on how you want to handle models without explicit chat templates.
            # For now, we assume models you add will generally have one.
            # A simple concatenation might work for some, but it's not robust.
            print(f"Warning: Tokenizer for {model_name} does not have a chat_template. Attempting manual formatting.")
            if "gemma" in repo_id: # Re-check for Gemma if no template
                 prompt_text = f"<start_of_turn>user\n{messages[0]['content']}<end_of_turn>\n<start_of_turn>model\n"
            else: # Generic attempt for others, might need per-model handling
                 prompt_text = messages[0]["content"] + "\n" + messages[1]["content"] if len(messages)>1 else messages[0]["content"]
            inputs_dict = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=context_length - max_tokens)
            input_ids = inputs_dict.input_ids.to(DEVICE)
            attention_mask = inputs_dict.attention_mask.to(DEVICE)
        else:
            try:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True, # Important for instruction-following models
                    return_tensors="pt"
                )
                input_ids = inputs.to(DEVICE)
                # Create attention mask (if not already part of 'inputs' from apply_chat_template, though it usually is for PT tensors)
                attention_mask = input_ids.ne(tokenizer.pad_token_id).to(DEVICE) if hasattr(input_ids, 'ne') else torch.ones_like(input_ids).to(DEVICE)

            except Exception as e_template:
                print(f"Error applying chat template for {model_name}: {e_template}")
                print("Falling back to basic tokenization without chat template.")
                # Construct a simple prompt string
                prompt_str = ""
                for msg in messages:
                    prompt_str += f"{msg['role']}: {msg['content']}\n" # Basic formatting
                if not "gemma" in repo_id: # Add assistant prompt for non-Gemma if needed
                    prompt_str += "assistant:"

                tokenized_inputs = tokenizer(prompt_str, return_tensors="pt", padding=True, truncation=True, max_length=context_length - max_tokens)
                input_ids = tokenized_inputs.input_ids.to(DEVICE)
                attention_mask = tokenized_inputs.attention_mask.to(DEVICE)


        print(f"Generating story with {model_name} in {language} (Target: ~{target_word_count} words, max_tokens={max_tokens}). Input shape: {input_ids.shape}")
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # Ensure EOS token is used for padding
        )
        # ... rest of your function

        # --- THIS IS THE PART TO REPLACE/UPDATE ---
        # Start of new logic for processing generated text:
        num_input_tokens = input_ids.shape[1]
        generated_token_ids = outputs[0][num_input_tokens:]
        story_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        # Specific cleanup for Gemma if it starts with "model\n"
        if "gemma" in repo_id:
            if story_text.lower().startswith("model\n"):
                story_text = story_text[len("model\n"):].lstrip()
            elif story_text.lower().startswith("model "): # Handle "model " if no newline
                story_text = story_text[len("model "):].lstrip()
        
        # You might want to add similar targeted cleanups for other models if they exhibit similar behavior
        # e.g., some models might start with their role name if not handled by skip_special_tokens

        if not story_text: # Check after initial strip and model-specific cleanup
            print("Warning: LLM returned empty response after decoding generated tokens.")
            story_text = f"Error: LLM returned an empty story in {language}."
            return story_text

        actual_words = len(story_text.split())
        print(f"Cleaned generated story: {actual_words} words, {len(story_text)} characters")
        if actual_words == 0 and not story_text.startswith("Error:"):
             print("Warning: Story text became empty after cleaning procedures.")
             story_text = f"Error: Story became empty after cleaning in {language}."
             return story_text
        # --- End of new logic for processing generated text ---

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

# ----> START MODS HERE - New function process_audio_only <----
def process_audio_only(story_text_from_box, language_selection, voice_choice, voice_sample_path, background_sound, temperature, speed):
    """Generates audio from the provided text using selected TTS and background settings."""
    status_text = ""
    default_sr = 24000
    audio_output_value = (default_sr, np.zeros(int(default_sr * 0.1), dtype=np.int16)) # Placeholder for errors
    download_file_path = None
    output_filename = "regenerated_audio.wav"
    output_dir = "output" # Ensure this directory exists or is created

    try:
        if not story_text_from_box or not story_text_from_box.strip():
            status_text = "Error: Story textbox is empty. Cannot generate audio.\n"
            yield status_text, None, None
            return

        status_text += f"Starting audio regeneration from textbox content...\n"
        status_text += f"Language: {language_selection}, Voice: {voice_choice}\n"
        yield status_text, None, None # Update status, no audio yet

        # 1. Generate Speech
        status_text += "Generating speech...\n"
        yield status_text, None, None
        
        actual_voice_sample_path = voice_sample_path if voice_sample_path and os.path.exists(str(voice_sample_path)) else None
        if voice_choice == "Clone from Sample" and not actual_voice_sample_path:
            status_text += "Warning: 'Clone from Sample' selected, but no valid voice sample provided. Attempting to use a default speaker if available or will error.\n"
            # TTS function will handle error if no speaker can be determined
        
        generated_speech_audio = generate_speech_tts(
            story_text_from_box, 
            voice_choice, 
            actual_voice_sample_path, 
            language=language_selection,
            temperature=temperature, 
            speed=speed
        )
        audio_output_value = generated_speech_audio

        actual_duration_seconds = 0
        if audio_output_value and audio_output_value[1] is not None and audio_output_value[1].size > 0:
            sample_rate, audio_data = audio_output_value
            actual_duration_seconds = len(audio_data) / sample_rate
            status_text += f"Speech generated ({actual_duration_seconds:.1f} seconds).\n"
        else:
            status_text += "Speech generation failed or produced empty audio.\n"
            # Ensure audio_output_value is the error placeholder if generation failed
            audio_output_value = (default_sr, np.zeros(int(default_sr * 0.1), dtype=np.int16))
        yield status_text, audio_output_value, None

        # 2. Mix with Background Sound
        status_text += "Mixing audio with background (if selected)...\n"
        yield status_text, audio_output_value, None
        
        final_audio_tuple = mix_audio_with_background(audio_output_value, background_sound)
        audio_output_value = final_audio_tuple
        status_text += "Audio mixing complete.\n"
        yield status_text, audio_output_value, None

        # 3. Save Audio and Provide Download
        download_file_path = None
        if audio_output_value:
            sample_rate, audio_data = audio_output_value
            if audio_data is not None and audio_data.size > 0:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                output_filename = f"regenerated_story_{timestamp}.wav" # Unique filename
                output_filepath = os.path.join(output_dir, output_filename)

                print(f"Saving regenerated audio to: {output_filepath}")
                try:
                    write_wav(output_filepath, sample_rate, audio_data)
                    status_text += f"Final audio saved: {output_filepath}\n"
                    download_file_path = output_filepath
                except Exception as write_e:
                    status_text += f"ERROR: Failed to write WAV file: {write_e}\n"
                    print(f"ERROR writing WAV: {write_e}")
                    download_file_path = None # Ensure it's None on error
            else:
                status_text += "Warning: Final audio data is empty or invalid after mixing. Skipping save.\n"
        else:
            status_text += "Warning: Audio processing failed. No file saved.\n"
        
        status_text += "Audio regeneration process finished.\n"
        yield status_text, audio_output_value, download_file_path

    except Exception as e:
        error_message = f"An unexpected error occurred during audio regeneration: {e}"
        print(error_message)
        import traceback
        traceback.print_exc()
        status_text += f"ERROR: {error_message}\n"
        # Ensure a safe audio output on error (e.g., short silence)
        error_audio_output = (default_sr, np.zeros(int(default_sr * 0.1), dtype=np.int16))
        yield status_text, error_audio_output, None

# ----> END MODS HERE - New function process_audio_only <----

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

# ----> START MODS HERE - Modified generate_story_audio function <----
def generate_story_audio(story_mode, # ---- NEW INPUT ----
                         prompt, 
                         manual_story_text, # ---- NEW INPUT ----
                         model_name_selection, 
                         language_selection, 
                         voice_choice, 
                         voice_sample_path, 
                         background_sound, 
                         target_minutes, 
                         temperature, 
                         speed):
    """
    Generates a story and converts it to audio based on the selected mode.
    If mode is 'Generate story with AI', it uses LLM.
    If mode is 'Use my own story text', it uses the provided manual_story_text.
    """
    status_text = ""
    story_text_to_output = "Error: Story processing did not complete." # This will hold the story text for the textbox
    default_sr = 24000
    audio_output_value = (default_sr, np.zeros(int(default_sr * 0.1), dtype=np.int16)) # Placeholder for errors
    download_file_path = None
    output_filename_prefix = "generated_story" # Prefix for the output file
    output_dir = "output" # Ensure this directory exists or is created

    try:
        status_text += f"Selected Mode: {story_mode}\n"
        yield status_text, story_text_to_output, None, None # Initial status update

        # --- Story Generation or Selection ---
        if story_mode == "Generate story with AI":
            status_text += f"Selected LLM: {model_name_selection}\n"
            status_text += f"Selected Language for Story: {language_selection}\n"
            
            WPM = 150 # Words Per Minute estimation
            target_word_count = int(target_minutes * WPM)
            # Ensure context length and token limits are respected (using existing logic from your file)
            selected_model_config = AVAILABLE_MODELS.get(model_name_selection, {})
            selected_model_context_length = selected_model_config.get("context_length", 4096)
            context_buffer = 512 
            max_allowed_tokens = max(selected_model_context_length - context_buffer, 512) 
            # Max tokens for LLM generation, e.g. 1800 or based on model limits.
            # You might want to adjust this based on typical story length in tokens vs words.
            # A rough estimate: 1 word ~ 1.33 tokens. So, target_word_count * 1.33 could be a guide.
            # Let's keep your existing max_tokens_for_llm logic or a fixed practical limit.
            max_tokens_for_llm = min(int(target_word_count * 2.0), max_allowed_tokens) # Allowing more tokens for generation
            max_tokens_for_llm = min(max_tokens_for_llm, 3000) # Cap at 3000 new tokens as a practical limit

            status_text += f"Generating story with AI (Target: ~{target_minutes:.1f} min / ~{target_word_count} words, max_tokens={max_tokens_for_llm})...\n"
            story_text_to_output = "AI is thinking..." # Placeholder while generating
            yield status_text, story_text_to_output, None, None

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
                story_text_to_output = error_msg # Update textbox with error
                yield status_text, story_text_to_output, None, None
                return

            story_text_to_output = generated_story_result # This is the AI-generated story
            actual_words = len(story_text_to_output.split())
            status_text += f"AI Story generated ({actual_words} words).\n"
            output_filename_prefix = f"ai_story_{model_name_selection.split(' ')[0].lower()}"

        elif story_mode == "Use my own story text":
            if not manual_story_text or not manual_story_text.strip():
                status_text += "Error: 'Use my own story text' mode selected, but the story textbox is empty. Please type or paste your story.\n"
                story_text_to_output = "Error: Story textbox is empty for 'Use my own story text' mode."
                yield status_text, story_text_to_output, None, None
                return
            
            story_text_to_output = manual_story_text # Use the text from the interactive textbox
            actual_words = len(story_text_to_output.split())
            status_text += f"Using user-provided story text ({actual_words} words).\n"
            output_filename_prefix = "user_story"
        
        else:
            status_text += f"Error: Unknown story mode selected: {story_mode}\n"
            story_text_to_output = f"Error: Unknown story mode: {story_mode}"
            yield status_text, story_text_to_output, None, None
            return

        yield status_text, story_text_to_output, None, None # Update textbox with final story text for this stage

        # --- Audio Generation (Common for both modes) ---
        status_text += f"Generating speech for the story in {language_selection}...\n"
        yield status_text, story_text_to_output, None, None # Update status before TTS

        actual_voice_sample_path = voice_sample_path if voice_sample_path and os.path.exists(str(voice_sample_path)) else None
        if voice_choice == "Clone from Sample" and not actual_voice_sample_path:
            status_text += "Warning: 'Clone from Sample' selected, but no valid voice sample provided. TTS may fail or use a default.\n"
            # The generate_speech_tts function should handle errors if no speaker can be determined.

        generated_speech_audio = generate_speech_tts(
            story_text_to_output, # Use the determined story text
            voice_choice, 
            actual_voice_sample_path, 
            language=language_selection, # language_selection is used for TTS
            temperature=temperature, 
            speed=speed
        )
        audio_output_value = generated_speech_audio

        actual_duration_seconds = 0
        if audio_output_value and audio_output_value[1] is not None and audio_output_value[1].size > 0:
            sample_rate, audio_data = audio_output_value
            actual_duration_seconds = len(audio_data) / sample_rate
            status_text += f"Speech generated ({actual_duration_seconds:.1f} seconds).\n"
        else:
            status_text += "Speech generation failed or produced empty audio.\n"
            audio_output_value = (default_sr, np.zeros(int(default_sr * 0.1), dtype=np.int16)) # Ensure error placeholder
        yield status_text, story_text_to_output, audio_output_value, None

        status_text += "Mixing audio with background (if selected)...\n"
        yield status_text, story_text_to_output, audio_output_value, None
        
        final_audio_tuple = mix_audio_with_background(audio_output_value, background_sound)
        audio_output_value = final_audio_tuple # Update with mixed audio (or original if no background)
        status_text += "Audio mixing complete.\n"
        yield status_text, story_text_to_output, audio_output_value, None

        # --- Save Audio and Provide Download (Common for both modes) ---
        download_file_path = None
        if audio_output_value:
            sample_rate, audio_data = audio_output_value
            if audio_data is not None and audio_data.size > 0:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                # Use the prefix determined by the mode
                final_output_filename = f"{output_filename_prefix}_{timestamp}.wav"
                output_filepath = os.path.join(output_dir, final_output_filename)

                print(f"Saving final audio to: {output_filepath}")
                try:
                    write_wav(output_filepath, sample_rate, audio_data)
                    story_word_count = len(story_text_to_output.split())
                    audio_duration_final = len(audio_data) / sample_rate if audio_data is not None else 0
                    status_text += f"Final audio saved ({story_word_count} words, {audio_duration_final:.1f}s): {output_filepath}\n"
                    download_file_path = output_filepath
                except Exception as write_e:
                    status_text += f"ERROR: Failed to write WAV file: {write_e}\n"
                    print(f"ERROR writing WAV: {write_e}")
                    download_file_path = None 
            else:
                status_text += "Warning: Final audio data is empty or invalid after mixing. Skipping save.\n"
        else:
            status_text += "Warning: Audio generation/processing failed. No file saved.\n"

        status_text += "Process finished.\n"
        # Final yield with all outputs
        yield status_text, story_text_to_output, audio_output_value, download_file_path

    except Exception as e:
        error_message = f"An unexpected error occurred in generate_story_audio: {e}"
        print(error_message)
        import traceback
        traceback.print_exc()
        # Determine story_text_for_error_yield safely
        final_story_text_for_error_yield = story_text_to_output if 'story_text_to_output' in locals() and isinstance(story_text_to_output, str) else "Error: Processing failed."
        if not isinstance(final_story_text_for_error_yield, str) or not final_story_text_for_error_yield.strip(): # ensure it's a non-empty string
            final_story_text_for_error_yield = f"Error during processing: {e}"

        status_text += f"ERROR: {error_message}\n"
        error_audio_output = (default_sr, np.zeros(int(default_sr * 0.1), dtype=np.int16)) # Safe audio output
        yield status_text, final_story_text_for_error_yield, error_audio_output, None
# ----> END MODS HERE - Modified generate_story_audio function <----

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

# ----> START MODS HERE - Function to update UI based on story mode <----
def update_ui_visibility_based_on_story_mode(story_mode):
    # Determine if AI-related inputs should be active
    ai_controls_active = (story_mode == "Generate story with AI")

    # Define the placeholder text for the prompt input based on mode
    prompt_placeholder_text = "e.g., A brave knight searching for a lost dragon..." \
        if ai_controls_active \
        else "Not used when 'Use my own story text' is selected. Type your story below."

    # Return gr.update directives for each component
    # The order here must match the 'outputs' list in the .change() listener
    return (
        gr.update(interactive=ai_controls_active, placeholder=prompt_placeholder_text),  # For prompt_input
        gr.update(interactive=ai_controls_active),  # For model_select_input
        gr.update(interactive=ai_controls_active)   # For duration_slider
    )
# ----> END MODS HERE - Function to update UI based on story mode <----

# ----> START MODS HERE - UI Elements <----
# (Existing code before this section remains the same)

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎙️ StoryCraft AI")
    gr.Markdown("Created by [TheAwakenOne](https://x.com/TheAwakenOne619)")
    # ----> START MODS HERE - How to Use Note <----
    with gr.Accordion("Quick Guide: How to Use StoryCraft AI", open=False): # Starts collapsed
        gr.Markdown(
            """
            **Welcome to StoryCraft AI! Here's a quick guide:**

            1.  **Choose Your Story Source (Section 1):**
                * **"Generate story with AI":** The app will create a new story based on your prompt, selected LLM, language, and desired duration.
                * **"Use my own story text":** Type or paste your complete story directly into the "Story Text (Editable)" box in Section 5. The AI story generation settings (prompt, LLM model, duration) will be ignored.

            2.  **Configure Settings (Sections 2, 3, 4):**
                * If generating with AI, fill in your "Story Prompt," choose an "LLM Model," and set the "Approx. Target Audio Duration."
                * Always choose your "Language (for Story & TTS)" – this is crucial for both AI story quality and accurate voice narration.
                * Select your desired "Voice Choice," upload a "Voice Sample" if cloning, and pick "Background Sound."
                * Fine-tune "TTS Temperature" and "Speed" for the narration.

            3.  **Generate!**
                * Click **"Generate Story & Audio" (Section 4):**
                    * In AI mode, this creates a *new* story from your prompt and then generates audio.
                    * In "Use my own story" mode, this generates audio directly from the text you entered/edited in the "Story Text" box.
                * The generated story (if AI) or your input story will appear in the "Story Text (Editable)" box (Section 5).

            4.  **Edit & Regenerate Audio (Section 5):**
                * You can freely edit the text in the "Story Text (Editable)" box.
                * To get new audio for your *edited text*, click the **"Regenerate Audio from Textbox"** button. This will *not* re-run the AI story generation.

            **Tips:**
            * The "Status Log" shows what the app is doing.
            * Ensure your chosen "Language" matches the language of the text for best TTS results.
            * For voice cloning, a clear WAV or MP3 sample works best.
            """
        )
    # ----> END MODS HERE - How to Use Note <----
    gr.Markdown( # Your existing markdown
        "This app was created as a hobby project. "
        "By using this app, you agree to the [Coqui Public Model License (CPML)](https://coqui.ai/cpml)."
    )
    gr.Markdown("Enter a prompt to generate a story in your chosen language, select a voice and background sound, and generate audio narration! You can also type/edit your own story for narration.")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 1. Story Input Mode")
            story_mode_input = gr.Radio(
                label="Choose Story Source",
                choices=["Generate story with AI", "Use my own story text"],
                value="Generate story with AI"
            )
            gr.Markdown("### 2. Story & Audio Generation Settings")
            prompt_input = gr.Textbox(lines=5, label="Story Prompt (if AI generating)", placeholder="e.g., A brave knight searching for a lost dragon...")
            model_select_input = gr.Dropdown(label="Choose LLM Model (if AI generating)", choices=model_choices, value=DEFAULT_MODEL_NAME)
            language_select_input = gr.Dropdown(label="Choose Language (for Story & TTS)", choices=language_choices, value=DEFAULT_LANGUAGE)
            
            gr.Markdown("### 3. Voice & Sound Settings")
            voice_choice_input = gr.Dropdown(label="Voice Choice", choices=voice_options, value="Clone from Sample")
            voice_sample_input = gr.Audio(label="Voice Sample for Cloning (Optional WAV/MP3)", type="filepath")
            background_sound_input = gr.Dropdown(label="Background Sound", choices=background_sound_options, value="None")
            
            gr.Markdown("### 4. TTS Tuning (Optional)")
            duration_slider = gr.Slider(minimum=1.0, maximum=5.0, step=0.5, value=3.0, label="Approx. Target Audio Duration (Minutes, for AI Story)")
            temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.65, label="TTS Temperature (Higher=More Random)")
            speed_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="TTS Speed (1.0 = Normal)")
            
            generate_button = gr.Button("Generate Story & Audio", variant="primary")

        with gr.Column(scale=3):
            status_textbox = gr.Textbox(label="Status Log", interactive=False, lines=5, placeholder="Status updates will appear here...")
            
            gr.Markdown("### Story Output & Editing")
            story_output_textbox = gr.Textbox(
                label="Story Text (Editable)", 
                interactive=True,  # ---- MODIFIED ----
                lines=10, 
                placeholder="AI-generated story will appear here, or you can type/paste your own."
            )
            
            gr.Markdown("### Audio Output")
            audio_output = gr.Audio(label="Generated Audio Output", type="filepath")
            regenerate_audio_button = gr.Button("Regenerate Audio from Textbox") # ---- NEW ----
            download_output = gr.File(label="Download Generated Audio")

     # --- Event Listeners ---
    # ----> START MODS HERE - Corrected placement for .change() listener <----
    story_mode_input.change(
        fn=update_ui_visibility_based_on_story_mode, # Make sure this function is defined globally (outside gr.Blocks)
        inputs=[story_mode_input],
        outputs=[
            prompt_input,
            model_select_input,
            duration_slider
        ]
    )
    # ----> Corrected placement for .change() listener <----       

    # ----> END MODS HERE - UI Elements <----

    # ----> START MODS HERE - Button Click Handlers <----

    # Updated click handler for the main "Generate Story & Audio" button
    generate_button.click(
        fn=generate_story_audio,
        inputs=[
            story_mode_input,         # New input: To know if AI gen or user text
            prompt_input,
            story_output_textbox,     # New input: For "Use my own story text" mode
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

    # New click handler for the "Regenerate Audio from Textbox" button
    regenerate_audio_button.click(
        fn=process_audio_only,
        inputs=[
            story_output_textbox,     # The story text to use
            language_select_input,    # Language for TTS
            voice_choice_input,
            voice_sample_input,
            background_sound_input,
            temperature_slider,
            speed_slider
        ],
        outputs=[status_textbox, audio_output, download_output] # Does not modify story_output_textbox
    )
    # ----> END MODS HERE - Button Click Handlers <----
    app.unload(unload_llm_model)

if __name__ == "__main__":
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    app.launch(debug=True)