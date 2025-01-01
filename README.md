# Fine-Tuning BERT for Phishing URL Classification

This project demonstrates how to fine-tune a pre-trained BERT model for the task of classifying phishing URLs. By leveraging the power of BERT, we aim to accurately distinguish between safe and phishing URLs. The project uses the `bert-base-uncased` model from Hugging Face's Transformers library and fine-tunes it on a phishing URL dataset.

## Table of Contents
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgements](#acknowledgements)


## Dataset

The dataset used in this project is a phishing URL dataset, which contains URLs labeled as either phishing or safe.

- **Source**: [shawhin/phishing-site-classification](https://huggingface.co/datasets/shawhin/phishing-site-classification)

## Model

The model used is `bert-base-uncased`, a pre-trained BERT model. The model is fine-tuned for sequence classification to distinguish between phishing and safe URLs.

## Training

The training process involves the following steps:

1. **Data Loading**: Load the phishing URL dataset.
2. **Tokenization**: Tokenize the text data using the `bert-base-uncased` tokenizer.
3. **Model Preparation**: Load the pre-trained `bert-base-uncased` model and configure it for sequence classification.
4. **Training**: Train the model using the `Trainer` class from the Transformers library.

## Evaluation

The model is evaluated on the validation set using accuracy and AUC as the metrics. The evaluation function computes the accuracy and AUC of the model's predictions.

## Results

The trained model achieves the following results on the validation set:

- **Accuracy**: 89.3%
- **AUC**: 94.9%

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Phishing URL Dataset](https://huggingface.co/datasets/shawhin/phishing-site-classification)

This project was developed as part of a learning exercise in fine-tuning language models for classification tasks.
