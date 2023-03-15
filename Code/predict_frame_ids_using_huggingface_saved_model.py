"""Predict frame ids using Hugging Face models."""
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
import torch
from pickle import load


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_data(data):
    """
    Tokenize data using a pretrained tokenizer.
    
    Args:
    data: Data in Huggingface format.

    Returns:
    tokenized_data: Tokenized data
    """
    return tokenizer(data["text"], truncation=True, padding=True, max_length=128)


def load_object_from_pickle(pickle_file):
    """
    Load a python object from a pickle file.

    Args:
    pickle_file: Enter the input pickle file path

    Returns:
    data_object: Data object stored in the pickle file
    """
    with open(pickle_file, 'rb') as pickle_load:
        data_object = load(pickle_load)
        return data_object


def write_lines_to_file(lines, file_path):
    """
    Write lines to a file.

    Args:
    lines: Lines to be written to the file.
    file_path: Enter the output file path.

    Returns: None
    """
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


def main():
    """
    Pass arguments and call functions here.
    
    Args: None

    Returns: None
    """
    parser = ArgumentParser(description='This program is about predicting the frame ids using a saved Hugging Face model.')
    parser.add_argument('--test', dest='te', help='Enter the test data in CSV format.')
    parser.add_argument('--i2l', dest='i2l', help='Enter the index to label pickle file.')
    parser.add_argument('--model', dest='mod', help='Enter the model directory.')
    parser.add_argument('--output', dest='out', help='Enter the output file path for predictions.')
    args = parser.parse_args()
    # model path is a directory in Huggingface
    loaded_model = AutoModelForSequenceClassification.from_pretrained(args.mod)
    index_to_label_dict = load_object_from_pickle(args.i2l)
    test_dataset =  load_dataset('csv', data_files={'test': args.te}, split='test')
    # 2 ways to predict: 1 with pipeline, the other being passing inputs to the model
    pipe = TextClassificationPipeline(model=loaded_model, tokenizer=tokenizer)
    # print the outputs on the test dataset
    predictions = pipe(test_dataset['text'])
    actual_labels = []
    for prediction in predictions:
        pred_label = prediction['label']
        pred_index = int(pred_label.split('_')[1])
        pred_actual_label = index_to_label_dict[pred_index]
        actual_labels.append(pred_actual_label)
    write_lines_to_file(actual_labels, args.out)


if __name__ == '__main__':
    main()
