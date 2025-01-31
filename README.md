# Translator

This repository contains a language translator application. The user interface is built with Streamlit, and it uses the T5 model called `google/mt5-small` for translation.

## Model

The translation model used in this application is `google/mt5-small`, a multilingual variant of the T5 model. T5, or Text-To-Text Transfer Transformer, is a transformer model that converts all NLP tasks into a text-to-text format. This allows the same model, loss function, and hyperparameters to be used across different tasks.

You can find more information about the T5 model in the [T5 paper](https://arxiv.org/abs/1910.10683).

The `google/mt5-small` model is available on Hugging Face: [google/mt5-small](https://huggingface.co/google/mt5-small).

## User Interface

The user interface for this translator is built using Streamlit, which allows for easy and interactive web applications.

## Usage

To run the application, follow these steps:
1. Clone the repository.
2. Install the required dependencies.
3. Run the Streamlit application.

```bash
git clone <repository-url>
cd Translator
pip install -r requirements.txt
streamlit run app.py
```
