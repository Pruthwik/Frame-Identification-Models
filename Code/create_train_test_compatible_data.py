"""Create train and test compatible data for Huggin Face."""
from datasets import Dataset
from datasets import load_dataset
from argparse import ArgumentParser
from pickle import dump
from datasets import ClassLabel


def read_lines_from_file(file_path):
    """
    Read lines from a file.

    Args:
    file_path: Enter the input file path.

    Returns:
    lines: Lines read from the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file_read:
        lines = [line.strip() for line in file_read.readlines() if line.strip()]
        return lines


def create_label_to_index_dict(labels):
    """
    Create a label to index dict from a list of labels.

    Args:
    labels: List of labels.

    Returns:
    dict_label_to_index: Dictionary from label to index.
    """
    return {label: index for (index, label) in enumerate(labels)}


def create_index_to_label_dict(labels):
    """
    Create an index to label dict from a list of labels.

    Args:
    labels: List of labels.

    Returns:
    dict_index_to_label: Dictionary from index to label.
    """
    return {index: label for (index, label) in enumerate(labels)}



def dump_object_into_pickle(data_object, pickle_file):
    """
    Dump a python object into a pickle file.

    Args:
    data_object: Data object to be pickled.
    pickle_file: Enter the path of the pickle file.

    Returns: None
    """
    with open(pickle_file, 'wb') as file_dump:
        dump(data_object, file_dump)


def main():
    """
    Pass arguments and call functions here.

    Args: None

    Returns: None
    """
    parser = ArgumentParser(description='This is a program to create compatible data for HuggingFace.')
    parser.add_argument('--train', dest='tr', help='Enter the train tsv file.')
    parser.add_argument('--test', dest='te', help='Enter the test tsv file.')
    parser.add_argument('--train_out', dest='tr_o', help='Enter the train file in Hugging Face format.')
    parser.add_argument('--test_out', dest='te_o', help='Enter the test file in Hugging Face format.')
    parser.add_argument('--i2l', dest='i2l', help='Enter the index to label pickle file.')
    args = parser.parse_args()
    train_dataset = Dataset.from_csv(args.tr, split='train', delimiter='\t', header='infer')
    print(train_dataset)
    all_labels = train_dataset['label']
    all_labels = sorted(set(all_labels))
    label_to_index_dict = create_label_to_index_dict(all_labels)
    index_to_label_dict = create_index_to_label_dict(all_labels)
    class_labels = ClassLabel(names=all_labels)
    train_dataset_features = train_dataset.features.copy()
    train_dataset_features['label'] = class_labels
    train_dataset = train_dataset.cast(train_dataset_features)
    train_dataset_new = train_dataset.align_labels_with_mapping(label_to_index_dict, 'label')
    test_dataset = Dataset.from_csv(args.te, split='test', delimiter='\t', header='infer')
    test_dataset_features = test_dataset.features.copy()
    class_labels_test = ClassLabel(names=all_labels)
    test_dataset_features['label'] = class_labels_test
    test_dataset = test_dataset.cast(test_dataset_features)
    test_dataset_new = test_dataset.align_labels_with_mapping(label_to_index_dict, 'label')
    train_dataset_new.to_csv(args.tr_o)
    test_dataset_new.to_csv(args.te_o)
    dump_object_into_pickle(index_to_label_dict, args.i2l)


if __name__ == '__main__':
    main()
