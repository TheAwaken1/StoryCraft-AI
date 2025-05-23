# StoryCraft AI

<p align="center">
  <img src="icon.png" alt="StoryCraft AI Logo" width="50%"/>
</p>

<a href="https://buymeacoffee.com/cmpcreativn" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="217" height="60">
</a>

Your Engaging Stories, Now Editable! Generate engaging 1 to 5-minute short stories with LLMs and convert them to audio with Coqui TTS, supports voice cloning, built in speakers and multilingual.

Hey Storytellers!

I'm absolutely thrilled to announce a massive update to the app, packed with features you've been asking for and some powerful new additions to supercharge your creativity! I've been working hard to give you even more control and flexibility. Here’s what’s new:

🚀 1. Explore Even More Creative AI Brains!
I've significantly upgraded the AI engine by adding a selection of new, powerful language models! This means more diverse writing styles, more sophisticated story generation, and even more creative possibilities for you to explore. You can now experiment with:
* google/gemma-2-2b-it
* qwen2.5-14b-instruct
* Nous Hermes 2 Pro llama 3
* llama 3 8b instruct
Dive in and see what new kinds of stories you can create!

✍️ 2. Write or Paste Your Own Stories:
Have a story you've already penned or an idea you're ready to type out? Now you can! The story area in the app, where you see the generated text, is now a fully interactive text box. Simply type or paste your complete story directly into it, and then use all the great audio generation features (like Coqui TTS, voice cloning, and multilingual support) to bring your personal narrative to life.

✂️ 3. Edit AI-Generated Stories to Perfection:
Love the creative spark the AI provides but want to fine-tune the details? Go for it! Any story generated by the AI can now be directly edited within that same text box. This means you can correct grammar, adjust plot points, change dialogue, remove sections, or add your unique flair until it's exactly how you envision it. The story box is your canvas!

🔄 4. Easily Switch Between AI Creation and Your Text:
I've added a simple new option right in the interface (you'll see choices like "Generate story with AI" or "Use my own story text"). This lets you seamlessly decide whether you want the AI to create a story based on your prompt, or if you want to use the text you've entered or edited in the story box. The app will intelligently use the correct source based on your selection.

🔊 5. Generate Audio from Your Final Masterpiece:
Once your story is perfect – whether it was initially AI-generated and you've edited it, or it’s entirely your own creation from scratch – getting the audio is just as easy as before. Just click the "Generate Audio" button (it might also say "Regenerate Audio" if you're making changes). The app will use the final text currently in the story box to produce your high-quality audio narration.

What This Means For You:

More AI Power: Access to a wider range of advanced AI models for richer story generation.
Total Creative Control: You're no longer just a prompter; you're the author and editor with the final say.
Ultimate Flexibility: Use the AI for a quick start, as a collaborator for ideas, or simply leverage the advanced audio engine for your completely original work.
Deeply Personalized Narratives: Tailor every story, AI-assisted or not, to be uniquely yours.
I'm incredibly excited to see how you use these new features to create even more amazing and engaging audio stories!

Thanks so much for your support! If you enjoy these updates and would like to see more apps and features like this in the future, please consider supporting my work. It truly helps me keep building and improving.

Happy Storytelling!

TheAwakenOne

License Notice
This app was created as a hobby project. By using this app, you agree to the Coqui Public Model License [CPML](https://coqui.ai/cpml).

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/TheAwaken1/StoryCraft-AI.git
    cd StoryCraft-AI
    ```

2. **Create a Virtual Environment:**

    On Windows:
    ```bash
    python -m venv env
    env\Scripts\activate
    ```

   On Mac/Linux:
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

    * (Using python or python3 depends on your system's Python installation.)
        Upgrade Pip and Install Dependencies:

3.  **Upgrade Pip and Install Dependencies:**
      ```bash
      python -m pip install --upgrade pip
      ```
      
 **Install Core AI Libraries:**
    * **`bitsandbytes`**: Required for efficient model loading (provides 4-bit quantization on NVIDIA GPUs).
  ```bash
  python -m pip install bitsandbytes>=0.43.0
  ```
        
  *(Note: 4-bit quantization benefits are primarily for NVIDIA GPUs. On other platforms like Mac or CPU-only, the model will load in 16-bit.)*

**`Coqui TTS`**: For text-to-speech functionality.**
  ```bash
  python -m pip install coqui-tts==0.25.3
  ```

3.  **Install Remaining Application Dependencies:**
    ```bash
    python -m pip install -r requirements.txt --no-cache-dir
    ```
    

4.  **Install PyTorch:** 
    *The version of PyTorch you install depends on your system (NVIDIA GPU, Mac, or CPU-only Linux/Windows). Coqui TTS also installs a *version of torch and torchaudio.

    * **For GPU (NVIDIA CUDA - Windows/Linux):**
        *To enable GPU acceleration, you need a PyTorch build compatible with your NVIDIA driver and installed CUDA Toolkit.*

        1. *Ensure you have the NVIDIA CUDA Toolkit installed (e.g., 11.8, 12.1, 12.4). You can find it on the NVIDIA Developer        website.*
        2. *Visit the official PyTorch website to find the correct pip install command for your specific CUDA version.*

        3. *Example for CUDA 12.1 (ensure your installed toolkit matches):*

        ```bash
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

        4. *Example for CUDA 12.1 or later:*

        ```bash
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
        ```
If you installed the wrong PyTorch, you might need to uninstall it first (python -m pip uninstall torch torchvision torchaudio) before running the CUDA-specific command to ensure the correct version is active.

## For Mac (Apple Silicon - MPS or CPU):
    *PyTorch will use MPS for acceleration on M1/M2 chips if available, otherwise CPU. The models will run in 16-bit precision (requiring more RAM).*

    ```bash
    python -m pip install torch torchvision torchaudio
    ```

## For CPU Only (Windows, Linux):
    *If you need torchvision (which is not strictly required by this app but often bundled), you can install it:*

    ```bash
    python -m pip install torch torchvision torchaudio
    ```

5. **Download NLTK Punkt Data:**
    *This is used for sentence tokenization.*

    ```bash
    python -c "import nltk, os; nltk_data_path = os.path.join(os.getcwd(), 'nltk_data'); os.makedirs(nltk_data_path, exist_ok=True); nltk.data.path.append(nltk_data_path); nltk.download('punkt', download_dir=nltk_data_path, quiet=True); print('NLTK punkt downloaded to', nltk_data_path)"
    ```

Set Coqui TTS License Environment Variable:

  On Windows (Command Prompt):
    
  ```bash
  set COQUI_TTS_AGREE_LICENSE=1
   ```

  On Windows (PowerShell):
    
  ```bash
  $env:COQUI_TTS_AGREE_LICENSE = "1"
   ```

  On Mac/Linux:
    
  ```bash
   export COQUI_TTS_AGREE_LICENSE=1
   ```
*You might want to add this to your shell's profile file (e.g., .bashrc, .zshrc, or manage environment variables through Windows System Properties) for persistence.*

Run the Application:

  ```bash
   python app.py
   ```

Open http://127.0.0.1:7860 in your web browser to access the Gradio UI.


Usage

Open the Gradio UI at http://127.0.0.1:7860.
Enter a story prompt (e.g., "A brave knight searching for a lost dragon").

Select an LLM:
    Phi-3 Mini: Faster, lower resource.
    Qwen 2 7B: More capable, higher resource.
Choose a voice (built-in or clone from an uploaded sample).
Select a background sound (optional).
Set the approximate target audio duration (up to 5 minutes).
Click "Generate Story & Audio".
Models are downloaded on the first run and cached in the models/ directory (e.g., ~2GB for Phi-3 Mini 4-bit, ~4GB for Qwen 2 7B 4-bit; larger if not quantized on Mac/CPU).

Requirements

Operating System: Windows, Mac, or Linux.
Hardware:
    -Windows/Linux with NVIDIA GPU:
        -NVIDIA GPU that supports CUDA (e.g., Maxwell architecture or newer; includes GTX 900 series, RTX series, 50-series, etc.).
        -VRAM (for 4-bit quantized models):
            -Phi-3 Mini: ~2-3GB VRAM recommended.
            -Qwen 2 7B: ~4-5GB VRAM recommended.
        -CUDA Toolkit compatible with your PyTorch installation (see PyTorch installation section).

Mac (Apple Silicon M1/M2/M3 or Intel):
    -Models will run on MPS (Apple Silicon) or CPU.
    -RAM (for 16-bit models, as 4-bit quantization with bitsandbytes is not used):
        -Phi-3 Mini: ~4-5GB RAM.
        -Qwen 2 7B: ~14-16GB RAM.
    -Performance will be slower than GPU-accelerated 4-bit models.

CPU Only (Windows/Linux):
    -Similar RAM requirements as Mac for 16-bit models.
    -Performance will be the slowest.


Software:
Python 3.10 recommended (pyenv or brew install python@3.10 on Mac).
ffmpeg for audio processing:
    -Windows: Install from ffmpeg.org (download binaries, extract, and add the bin folder to your system's PATH) or via a package manager like Chocolatey (choco install ffmpeg).
    -Mac: brew install ffmpeg.
    -Linux: sudo apt-get install ffmpeg (Ubuntu/Debian) or equivalent for your distribution.

Models:
Phi-3-mini-4k-instruct (~4GB VRAM, fast).
Qwen2-7B-Instruct (~6GB VRAM, balanced).


Disk Space:
    ~500MB for dependencies.
    Models (downloaded on first run to models/ directory):
        Phi-3-mini-4k-instruct: ~2.2GB (HF fp16 version, loaded as ~2GB in 4-bit on NVIDIA, ~4GB in 16-bit on Mac/CPU).
        Qwen2-7B-Instruct: ~14GB (HF fp16 version, loaded as ~4GB in 4-bit on NVIDIA, ~14GB in 16-bit on Mac/CPU).
    Allow an additional few GB for Hugging Face cache if models are downloaded via snapshot_download's caching mechanism (by default, it uses cache_dir inside the repo if not globally configured).

Models Used
    LLMs (loaded with 4-bit quantization on NVIDIA GPUs, 16-bit on Mac/CPU):
        microsoft/Phi-3-mini-4k-instruct
        Qwen/Qwen2-7B-Instruct
TTS:
    tts_models/multilingual/multi-dataset/xtts_v2 (Coqui TTS)

Notes
The app uses Hugging Face Transformers. On NVIDIA GPUs, models are loaded in 4-bit for reduced VRAM usage. On Mac (MPS/CPU) and CPU-only systems, models are loaded in 16-bit precision, requiring more RAM.
NLTK punkt data for sentence tokenization is downloaded to the nltk_data/ directory within the repository.
LLM and TTS models are downloaded on their first use and cached (LLMs in models/, Coqui TTS models typically in ~/.local/share/tts/).
Setting the COQUI_TTS_AGREE_LICENSE=1 environment variable bypasses the Coqui TTS license prompt during TTS model loading. The app also attempts to agree programmatically.
Background audio files (motivational_theme.mp3, excitement_theme.mp3) are included in the backgrounds/ directory. You can replace these with your own MP3 files using the same filenames to customize.
Generated audio is saved in the output/ directory.

Troubleshooting

NLTK Punkt Not Found:
NLTK Punkt Not Found:

Re-run the NLTK download command from the installation steps.
Ensure you have an active internet connection.
Manually download punkt.zip from NLTK Data, create nltk_data/tokenizers/ directories in your project root, and extract punkt.zip into nltk_data/tokenizers/punkt/.


Manually download punkt.zip from https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip and extract to nltk_data/tokenizers/punkt.

Model Download Fails (LLM or TTS):
Check your internet connection.
Ensure you can access Hugging Face Hub and Coqui's model repositories.
Delete any partial downloads in models/ or ~/.local/share/tts/ and retry.
Firewall/VPN might sometimes interfere with downloads.

Audio Issues (No sound, errors during mixing):

Verify ffmpeg is installed correctly and added to your system's PATH. You can test by typing ffmpeg -version in your terminal.
Ensure background audio files exist in backgrounds/ if you selected one.
Check for errors in the console output from pydub or Coqui TTS.
Performance Issues / Out of Memory (OOM) Errors:

NVIDIA GPU:
Monitor VRAM usage with nvidia-smi (Linux/Windows).
If VRAM is an issue, use the Phi-3 Mini model. Ensure drivers are up to date.
Mac / CPU-only:
Remember that models run in 16-bit precision, requiring significant RAM (see Requirements section).
Phi-3 Mini is much more likely to run smoothly than Qwen 2 7B on systems with limited RAM (e.g., &lt;16-24GB for Qwen 2).
Close other resource-intensive applications.
On Mac, verify MPS is being used if available: python3 -c "import torch; print(torch.backends.mps.is_available())" (should print True).
CouldntDecodeError from pydub:

This almost always means ffmpeg is not found or not working correctly. Double-check its installation and that its bin directory is in your system's PATH. Restart your terminal/IDE after modifying PATH.

