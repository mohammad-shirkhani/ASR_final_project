## Full Fine-Tuning

The ASR model was fine-tuned using the full fine-tuning approach on our preprocessed Farsi YouTube 10-second dataset. Unlike partial or parameter-efficient fine-tuning methods, full fine-tuning updates all the parameters of the pre-trained SpeechT5 model, allowing it to fully adapt to the Farsi language characteristics.

The fine-tuned model can be accessed at:  
[https://huggingface.co/mohammad-shirkhani/speecht5_asr_finetune_persian](https://huggingface.co/mohammad-shirkhani/speecht5_asr_finetune_persian)

### Fine-Tuning Overview

- **Data**: The model was fine-tuned on a large-scale, preprocessed Farsi dataset consisting of 10-second audio clips.
- **Method**: Full fine-tuning was employed, ensuring that all layers of the SpeechT5 model are updated to capture the unique phonetic and linguistic patterns of Farsi.
- **Hyperparameters**: Key settings included a learning rate of 1e-4, batch size of 16 (with gradient accumulation), and 11,000 training steps, which provided a good balance between convergence and generalization.

This comprehensive fine-tuning approach has resulted in a model that achieves competitive ASR performance on Farsi speech.

