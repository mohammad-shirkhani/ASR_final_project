## Gradio Interface and Hugging Face Spaces Deployment

We developed a simple, user-friendly interface using Gradio to interact with our Farsi ASR system. The interface supports:
- **Audio File Upload**: Directly upload an audio file.
- **Video File Upload**: Upload a video file (audio is automatically extracted).
- **Live Recording**: Record your voice via the microphone.

The input is processed using our fine-tuned SpeechT5 model to generate transcriptions. The entire system is deployed on Hugging Face Spaces and can be accessed at:  
[https://huggingface.co/spaces/alibababeig/SpeechT5-Farsi-ASR](https://huggingface.co/spaces/alibababeig/SpeechT5-Farsi-ASR)

This deployment on CPU Spaces provides a public demo that automatically handles longer audio inputs by segmenting them into manageable chunks.

