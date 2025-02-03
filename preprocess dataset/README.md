## Preprocess Dataset

This section describes the preprocessing steps applied to the raw Farsi YouTube ASR dataset and provides links to the processed datasets available on Hugging Face.

### Processed Datasets

- **Training Dataset**:  
  [https://huggingface.co/datasets/mohammad-shirkhani/asr-farsi-youtube-chunked-10-seconds](https://huggingface.co/datasets/mohammad-shirkhani/asr-farsi-youtube-chunked-10-seconds)

- **Validation Dataset**:  
  [https://huggingface.co/datasets/mohammad-shirkhani/asr-farsi-youtube-chunked-10-seconds-val](https://huggingface.co/datasets/mohammad-shirkhani/asr-farsi-youtube-chunked-10-seconds-val)

- **Test Dataset**:  
  [https://huggingface.co/datasets/mohammad-shirkhani/asr-farsi-youtube-chunked-10-seconds-test](https://huggingface.co/datasets/mohammad-shirkhani/asr-farsi-youtube-chunked-10-seconds-test)

### Preprocessing Overview

To prepare the dataset for training our ASR system, several preprocessing steps were applied:

1. **Resampling**:  
   All audio files were resampled to a fixed sampling rate of 16 kHz to ensure consistency across the dataset.

2. **Noise Reduction**:  
   Using the `noisereduce` library, spectral gating techniques were applied to each audio segment to minimize background noise. This step helps in improving the clarity of the speech signals.

3. **Text Normalization**:  
   The transcriptions were cleaned and standardized using the Hazm library. This involved normalizing Persian text, mapping variant characters to a unified form, and removing unwanted punctuation and extra spaces.

4. **Chunking**:  
   Audio files were divided into 10-second segments to facilitate efficient processing with our encoder-decoder model. This chunking helps manage memory usage and improves alignment between audio and text during training.

5. **Dataset Structuring**:  
   Finally, processed audio files (saved in MP3 format) were paired with their corresponding normalized transcriptions and uploaded as separate datasets for training, validation, and testing on Hugging Face.

These preprocessing steps ensure that the data is in a consistent, high-quality format, making it well-suited for fine-tuning our ASR model.

