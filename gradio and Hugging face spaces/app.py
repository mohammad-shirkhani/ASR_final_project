import re
import time

import ffmpeg
import gradio as gr
import librosa
import noisereduce as nr
import numpy as np
from transformers import (
    SpeechT5Processor,
    SpeechT5ForSpeechToText,
)


HF_MODEL_PATH = 'mohammad-shirkhani/speecht5_asr_finetune_persian'

AUDIO_SAMPLING_RATE = 16000  # Hz

AUDIO_FRAME_MIN_DUR = 4.5    # second
AUDIO_FRAME_MAX_DUR = 11.5   # second

SILENCE_FRAME_DUR = 0.300    # second
SILENCE_FRAME_SHIFT = 0.010  # second

TEXT_GEN_MAX_LEN = 250       # character

model = None
processor = None


def initialize_model():
    global model
    global processor

    model = SpeechT5ForSpeechToText.from_pretrained(HF_MODEL_PATH)
    processor = SpeechT5Processor.from_pretrained(HF_MODEL_PATH)


def handle_user_input(audio_path, video_path):
    t_start = time.time()
    audio_asr_result = None
    video_asr_result = None

    if audio_path is not None:
        # Load the uploaded audio file and resample to 16 KHz
        waveform, sample_rate = librosa.load(audio_path, sr=None)
        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=AUDIO_SAMPLING_RATE
        )

        # Perform ASR on the audio waveform
        audio_asr_result = perform_asr(waveform)

    if video_path is not None:
        # Load the uploaded video file and extract its audio
        (
            ffmpeg
            .input(video_path)
            .output('tmp.wav', acodec='pcm_s16le')
            .run(overwrite_output=True)
        )

        # Load the extracted audio file and resample to 16 KHz
        waveform, sample_rate = librosa.load('tmp.wav', sr=None)
        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=AUDIO_SAMPLING_RATE
        )

        # Perform ASR on the audio waveform
        video_asr_result = perform_asr(waveform)

    delta_t = time.time() - t_start
    print(f'Total Time      = {delta_t:5.1f} s\n')

    return audio_asr_result, video_asr_result


def perform_asr(waveform):
    # Mono, nothing to be done :)
    if waveform.ndim == 1:
        pass
    # Stereo, convert to mono by averaging the channels
    elif waveform.ndim == 2 and waveform.shape[1] == 2:
        waveform = np.mean(waveform, axis=1)
    else:
        raise ValueError(f'Bad audio array shape: "{waveform.shape}"')

    t_start = time.time()
    # Split the audio array into smaller frames
    audio_frames = []
    start_idx = 0
    while start_idx != len(waveform):
        frame_end_min = int(
            start_idx + AUDIO_FRAME_MIN_DUR * AUDIO_SAMPLING_RATE
        )
        frame_end_max = int(
            start_idx + AUDIO_FRAME_MAX_DUR * AUDIO_SAMPLING_RATE
        )

        if frame_end_max < len(waveform):
            break_point = search_for_breakpoint(
                waveform,
                frame_end_min,
                frame_end_max
            )
        else:
            break_point = len(waveform)

        audio_frames.append(waveform[start_idx:break_point])
        start_idx = break_point

    delta_t = time.time() - t_start
    print(f'Audio Framing   = {delta_t:5.1f} s')

    t_start = time.time()
    # Apply noise reduction on each audio frame
    audio_frames = [
        nr.reduce_noise(y=frame, sr=AUDIO_SAMPLING_RATE)
        for frame in audio_frames
    ]
    delta_t = time.time() - t_start
    print(f'Noise Reduction = {delta_t:5.1f} s')

    ######################### Method 1 - For Loop #########################

    # transcriptions = []
    # for frame in audio_frames:
    #     inputs = processor(
    #         audio=frame,
    #         sampling_rate=AUDIO_SAMPLING_RATE,
    #         return_tensors='pt'
    #     )
    #     predicted_ids = model.generate(
    #         **inputs,
    #         max_length=TEXT_GEN_MAX_LEN
    #     )
    #     transcription = processor.batch_decode(
    #         predicted_ids,
    #         skip_special_tokens=True
    #     )[0]

    #     transcriptions.append(transcription)

    ######################### Method 2 - Batch ############################

    t_start = time.time()
    # Process the entire batch of audio frames
    inputs = processor(
        audio=audio_frames,
        sampling_rate=AUDIO_SAMPLING_RATE,
        padding=True,
        return_tensors='pt'
    )

    # Generate predictions for the entire batch
    predicted_ids = model.generate(
        **inputs,
        max_length=TEXT_GEN_MAX_LEN
    )

    # Decode the predicted IDs into transcriptions
    transcriptions = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )
    delta_t = time.time() - t_start
    print(f'Text Generation = {delta_t:5.1f} s')

    t_start = time.time()
    # Clean the model-generated transcriptions
    transcriptions = [clean_model_answer(t) for t in transcriptions]
    delta_t = time.time() - t_start
    print(f'Text Cleaning   = {delta_t:5.1f} s')

    return '\n\n'.join(transcriptions)


def search_for_breakpoint(waveform, begin, end):
    waveform_ampl = np.abs(waveform)
    frame_size = int(SILENCE_FRAME_DUR * AUDIO_SAMPLING_RATE)
    frame_shift = int(SILENCE_FRAME_SHIFT * AUDIO_SAMPLING_RATE)

    avg_amplitudes = {}
    for start_idx in range(begin, end - frame_size + 1, frame_shift):
        stop_idx = start_idx + frame_size
        avg_amplitudes[start_idx] = np.mean(waveform_ampl[start_idx:stop_idx])

    # Consider the center of the most quiet frame as the breakpoint
    best_start_idx = min(avg_amplitudes, key=avg_amplitudes.get)
    break_point = best_start_idx + int(frame_size / 2)
    return break_point


def clean_model_answer(txt):
    txt = re.sub(r'\s(?!\s)', '', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt


if __name__ == '__main__':
    # Initialize the ASR model
    initialize_model()

    # Create a Gradio interface with required inputs and outputs
    iface = gr.Interface(
        fn=handle_user_input,
        inputs=[
            gr.Audio(label='Upload/Record Audio', type='filepath'),
            gr.Video(label='Upload Video', sources='upload'),
        ],
        outputs=[
            gr.Textbox(label="Audio Transcript", rtl=True),
            gr.Textbox(label="Video Transcript", rtl=True),
        ],
        title="Automatic Speech Recognition for Farsi Language",
        description="Upload an audio/video file to generate its transcript!",
        examples=[
            ['examples/roya_nonahali.mp3', None],         # Example Audio 1
            ['examples/keikavoos_yakideh.mp3', None],     # Example Audio 2
            ['examples/amirmohammad_samsami.mp3', None],  # Example Audio 3
        ],
        cache_examples=False,
    )
    # Launch the Gradio app
    iface.launch()
