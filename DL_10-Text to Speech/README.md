
# Assignment 10: Multilingual Audio Classification, Speech-to-Text and Text-to-Speech

## Overview

This assignment implements a multilingual speech processing system capable of identifying a spoken Indian language, transcribing the speech into text, and converting the transcription back into synthesized speech.

The application integrates three deep learning components:

1. Audio language classification using a CRNN.
2. Speech-to-text using OpenAI Whisper.
3. Text-to-speech using Meta MMS-TTS.

The system supports Hindi, Marathi, Tamil, and Telugu.

## Objectives

- Understand deep learning for audio classification.
- Process raw audio signals.
- Extract log-mel spectrogram features.
- Implement a Convolutional Recurrent Neural Network.
- Perform multilingual speech recognition.
- Generate synthesized speech.
- Evaluate classification, transcription, and speech synthesis.
- Build an integrated speech processing pipeline.

## Dataset

**Google FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)**

Source: Hugging Face

```text
google/fleurs
```

The implementation uses speech samples from four Indian languages:

| Language | Dataset Code |
|---|---|
| Hindi | hi_in |
| Marathi | mr_in |
| Tamil | ta_in |
| Telugu | te_in |

The notebook selects 200 utterances per language from the FLEURS validation data, producing 800 selected audio samples.

Sentence-group-aware splitting is used to separate training, validation, and testing examples.

## Technologies Used

- Python
- PyTorch
- Torchaudio
- Hugging Face Transformers
- Hugging Face Hub
- Librosa
- SoundFile
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- JiWER
- Uroman

## System Architecture

```text
Input Speech
     |
     v
Audio Preprocessing
     |
     v
CRNN Language Classifier
     |
     v
Predicted Language
     |
     v
Whisper-Small Speech-to-Text
     |
     v
Transcribed Text
     |
     v
MMS-TTS / VITS
     |
     v
Synthesized Speech
```

## Component 1: Audio Classification

### Model: CRNN

A Convolutional Recurrent Neural Network combines convolutional layers with a bidirectional GRU.

### Feature Extraction

Audio is processed using:

- Sampling rate: 16,000 Hz
- Primary classification clip length: 4 seconds
- Log-mel spectrograms
- 64 mel frequency bands
- Frequency masking
- Time masking

### Architecture

The CRNN consists of:

1. Three convolutional blocks with 32, 64, and 128 output channels.
2. Batch normalization and max pooling.
3. Bidirectional GRU with hidden size 128.
4. Temporal feature aggregation.
5. Dropout and a fully connected classification layer.

The network predicts one of four supported languages.

### Training

| Parameter | Value |
|---|---|
| Epochs | 25 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Loss | CrossEntropyLoss |
| Label smoothing | 0.1 |
| Scheduler | CosineAnnealingLR |

The best model parameters are selected using validation macro-F1.

## Component 2: Speech-to-Text

### Model: Whisper-Small

Pretrained model:

```text
openai/whisper-small
```

Whisper uses a Transformer encoder-decoder architecture for speech recognition.

The predicted language from the CRNN is passed to Whisper to guide transcription.

### Evaluation Metrics

- Word Error Rate (WER)
- Character Error Rate (CER)

Lower error rates indicate better transcription performance.

The notebook evaluates selected held-out audio clips from each language against reference transcripts.

## Component 3: Text-to-Speech

### Model: MMS-TTS

The implementation uses Meta's multilingual speech synthesis models:

```text
facebook/mms-tts-hin
facebook/mms-tts-mar
facebook/mms-tts-tam
facebook/mms-tts-tel
```

These models use the VITS speech synthesis architecture.

The pipeline converts text into generated speech for the selected language.

The notebook also uses romanization where required by the tokenizer.

### Evaluation

The implementation measures:

- Synthetic audio duration
- Speech recognition error on synthesized speech
- Round-trip Character Error Rate (CER)

Round-trip evaluation transcribes synthesized speech using Whisper and compares the result against the reference text.

## Experimental Results

The saved notebook execution reports:

### Audio Classification

| Metric | Result |
|---|---:|
| Test accuracy | 0.9231 |
| Macro-F1 | 0.9215 |

### Speech-to-Text

| Metric | Result |
|---|---:|
| Average WER | 1.190 |
| Average CER | 0.929 |

### Text-to-Speech

| Metric | Result |
|---|---:|
| Round-trip CER on human speech | 0.385 |
| Round-trip CER on synthetic speech | 0.568 |

These are the recorded results for the evaluated samples. The speech recognition error rates indicate that transcription quality remains an area for improvement.

## Visualizations

The notebook includes:

- Language distribution charts
- Audio duration analysis
- Waveform visualizations
- Log-mel spectrograms
- Training loss curves
- Classification accuracy curves
- Confusion matrix
- Accuracy versus audio duration
- t-SNE visualization of CRNN embeddings
- WER and CER comparisons
- Human versus synthesized speech spectrograms
- Audio playback demonstrations

## Integrated Demonstration

The notebook implements a complete pipeline:

```python
def pipeline(wav):
    p = predict([wav])[0]
    code = CODES[p.argmax()]
    text = transcribe(wav, LANGS[code][1])
    return LANGS[code][0], p.max(), text, tts(text, code)
```

The pipeline returns:

1. Predicted language
2. Prediction confidence
3. Transcribed text
4. Generated speech waveform

A Tamil test sample is used to demonstrate the end-to-end pipeline.

## How to Run

1. Open `Assignment-10.ipynb` in Google Colab.
2. Enable a GPU runtime, preferably T4 or better.
3. Install the required dependencies:

```bash
pip install torch torchaudio transformers huggingface_hub librosa soundfile pandas numpy matplotlib seaborn scikit-learn jiwer uroman pyarrow
```

4. Execute the notebook cells sequentially.
5. Allow the dataset and pretrained models to download.
6. Review the evaluation results and audio demonstrations.

An internet connection is required to download the dataset and pretrained model weights.

## Learning Outcomes

- Understanding audio preprocessing and feature extraction.
- Implementing CRNNs for spoken language identification.
- Working with convolutional and recurrent architectures.
- Applying pretrained transformer models to speech recognition.
- Generating multilingual speech using VITS.
- Evaluating speech recognition using WER and CER.
- Understanding round-trip speech evaluation.
- Integrating multiple AI models into one application.

## Conclusion

This assignment demonstrates an integrated multilingual speech processing pipeline combining audio classification, speech recognition, and speech synthesis.

The CRNN achieves approximately 92.31% test accuracy in the recorded language classification experiment.

The Whisper and MMS-TTS components demonstrate multilingual transcription and speech generation, while the evaluation metrics highlight the challenges of achieving consistently accurate speech recognition.

Overall, the assignment provides practical experience integrating several deep learning architectures into a single audio processing application.

## Files

- `Assignment-10.ipynb` — Multilingual audio classification, STT, and TTS implementation
- `README.md` — Assignment documentation


