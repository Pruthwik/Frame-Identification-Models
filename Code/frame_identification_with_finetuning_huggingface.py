"""Fine tuning a frame identification model using Hugging Face base model."""
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import TextClassificationPipeline
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
    parser = ArgumentParser(description='This program is about finetuning a frame identification model.')
    parser.add_argument('--train', dest='tr', help='Enter the training data in CSV format.')
    parser.add_argument('--test', dest='te', help='Enter the test data in CSV format.')
    parser.add_argument('--i2l', dest='i2l', help='Enter the index to label pickle file.')
    parser.add_argument('--model', dest='mod', help='Enter the model directory.')
    parser.add_argument('--epoch', dest='ep', help='Enter the number of epochs.', type=int)
    parser.add_argument('--output', dest='out', help='Enter the output file path for predictions.')
    args = parser.parse_args()
    train_dataset =  load_dataset('csv', data_files={'train': args.tr}, split='train')
    test_dataset =  load_dataset('csv', data_files={'test': args.te}, split='test')
    index_to_label_dict = load_object_from_pickle(args.i2l)
    num_labels = len(index_to_label_dict)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    # create the tokenized dataset
    train_tokenized_dataset = train_dataset.map(tokenize_data, batched= True)
    test_tokenized_dataset = test_dataset.map(tokenize_data, batched= True)
    training_args = TrainingArguments(
        args.mod,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.ep,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=test_tokenized_dataset,
        tokenizer=tokenizer
    )
    # train a model with specified arguments
    trainer.train()
    # if the model is to be trained from the latest checkpoint
    # trainer.train(resume_from_checkpoint=args.mod)
    # to predict and return the class/label with the highest score
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    # print the outputs on the evaluation dataset
    print('Training Done')
    predictions = pipe(test_dataset['text'])
    # the below can be uncommented if you want to save the most recent model
    # model.save_pretrained(args.mod)
    actual_labels = []
    for prediction in predictions:
        pred_label = prediction['label']
        pred_index = int(pred_label.split('_')[1])
        pred_actual_label = index_to_label_dict[pred_index]
        actual_labels.append(pred_actual_label)
    write_lines_to_file(actual_labels, args.out)


if __name__ == '__main__':
    main()
