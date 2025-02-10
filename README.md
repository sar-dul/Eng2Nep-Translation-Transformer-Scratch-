
# English to Nepali Translation Transformer (Scratch)

This repository implements an English to Nepali translation model using a Transformer-based architecture. The tokenizer (Byte Pair Encoding - BPE) for both English and Nepali languages is trained from scratch. The model is based on the "Attention Is All You Need" paper and uses the English-to-Nepali translation dataset from Hugging Face.

## Project Overview

This project includes the following steps:
1. **Tokenizer Training**: Trained the BPE tokenizers for both English and Nepali languages from scratch.
2. **Encoder-Decoder Architecture**: Implemented a Transformer-based model for translation, inspired by the architecture described in the "Attention Is All You Need" paper.
3. **Training**: Trained the encoder-decoder model on the English-to-Nepali dataset.

## Files

- `tokenizer_training_and_saving.ipynb`: Code for training the BPE tokenizers for both English and Nepali languages and saving them for later use.
- `encoder_decoder.ipynb`: Code for creating and training the encoder-decoder transformer model for English to Nepali translation.

## Requirements

- Python 3.x
- TensorFlow or PyTorch
- Hugging Face Datasets
- NumPy
- Matplotlib (for visualization)

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/eng2nep-translation-transformer.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Training Tokenizer**:
   - Run `tokenizer_training_and_saving.ipynb` to train and save the English and Nepali tokenizers.
   
2. **Training Model**:
   - Run `encoder_decoder.ipynb` to train the translation model using the pre-trained tokenizers.
   
3. **Inference**:
   - After training, you can use the trained model to translate English sentences to Nepali.

## Model Details

The model uses an encoder-decoder architecture with self-attention mechanisms, allowing it to capture the relationships between words in a sequence regardless of their distance from each other. The BPE tokenizers break down words into subword units, ensuring that the model can handle out-of-vocabulary words effectively.

## Dataset

The English-to-Nepali translation dataset used in this project is sourced from Hugging Face's datasets library. The dataset contains parallel translations between English and Nepali sentences.

## Results

Upon training, the model can generate Nepali translations from English sentences. Evaluation metrics such as BLEU score can be used to assess translation quality.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The "Attention Is All You Need" paper for the Transformer architecture.
- Hugging Face for the English-to-Nepali translation dataset.
