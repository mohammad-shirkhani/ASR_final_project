# ASR_final_project

## Overview

This project implements a Farsi Automatic Speech Recognition (ASR) system using the SpeechT5 model. SpeechT5 is a unified-modal encoder-decoder architecture originally inspired by the T5 model in natural language processing. In this project, we extend its capabilities to process both speech and text inputs for various spoken language processing tasks, with a focus on ASR. Despite the model not being pre-trained on Farsi data, our fine-tuning process on a 10-second Farsi YouTube dataset has yielded promising results.

## Project Structure

- **Model and Training**:  
  - The project uses the [microsoft/speecht5_asr](https://huggingface.co/microsoft/speecht5_asr) model from Hugging Face.
  - We employed the Transformers library for full fine-tuning of the model on our Farsi dataset.
  - Key hyperparameters include a learning rate of 1e-4, a batch size of 16 (with gradient accumulation steps = 2), 500 warmup steps, and a maximum of 11,000 training steps.
  
- **Data Preparation**:  
  - The dataset is split into 10-second audio clips, which ensures efficient mini-batch processing and improved alignment.
  - Preprocessing includes resampling all audio to 16 kHz, noise reduction using spectral gating (via the `noisereduce` library), and text normalization with the Hazm library.
  - Processed datasets (train, validation, and test) are uploaded to Hugging Face Hub for easy accessibility.

- **Evaluation**:  
  - The model's performance is evaluated using the Word Error Rate (WER) and Character Error Rate (CER) metrics.
  - Our evaluation on the test set yielded WER ≈ 15.3% and CER ≈ 12.2%, demonstrating the system’s competitive performance in Farsi ASR.

- **User Interface & Deployment**:  
  - A user-friendly interface is implemented using Gradio, supporting three types of inputs:
    - **Audio File**: Upload an audio file.
    - **Video File**: Upload a video file (audio is extracted automatically).
    - **Live Recording**: Capture live audio from a microphone.
  - The final system is deployed on Hugging Face Spaces (CPU version), making it publicly accessible at:  
    [SpeechT5-Farsi-ASR on Hugging Face Spaces](https://huggingface.co/spaces/alibababeig/SpeechT5-Farsi-ASR)

## How It Works

1. **Input Handling**:  
   The system accepts audio and video inputs. For video inputs, the audio track is extracted using `ffmpeg`.

2. **Preprocessing Pipeline**:  
   - All audio inputs are resampled to 16 kHz.
   - Audio signals are split into ~10-second frames, ensuring efficient processing.
   - Noise reduction is applied to each frame using the `noisereduce` library.
   - The cleaned frames are passed through the SpeechT5 model to generate transcriptions.

3. **Postprocessing**:  
   A cleaning function refines the model’s output (e.g., removing extra spaces) to deliver a user-friendly transcript.

4. **Output**:  
   Transcriptions from both audio and video inputs are displayed in separate text boxes on the Gradio interface.

## Installation and Requirements

To run this project locally or in a Hugging Face Space, ensure you have the following libraries installed:

- `transformers`
- `datasets`
- `torchaudio`
- `soundfile`
- `speechbrain`
- `accelerate`
- `gradio`
- `ffmpeg-python`
- `librosa`
- `noisereduce`
- `evaluate`
- `jiwer`

You can install the required packages using:
```bash
pip install transformers datasets torchaudio soundfile speechbrain accelerate gradio ffmpeg-python librosa noisereduce evaluate jiwer
