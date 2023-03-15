"""Find precision, recall, and F1-scores using sklearn library."""
from argparse import ArgumentParser
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


def read_lines_from_file(file_path):
    """
    Read lines from a file.

    Args:
    file_path: Enter the input file path.

    Returns:
    lines: Lines read from file
    """
    with open(file_path, 'r', encoding='utf-8') as fileRead:
        lines = [line.strip() for line in fileRead.readlines() if line.strip()]
        return lines


def find_precision_recall_F1score(gold_labels, predicted_labels, true_labels=None):
    """
    Find precision, recall, and F1-scores.

    Args:
    gold_labels: Gold labels read from the gold file.
    predicted_labels: Predicted labels from the predicted file.
    true_labels: True labels for generating the performance scores.
    
    Returns:
    report: Report with all the scores.
    """
    report = classification_report(gold_labels,
                                 predicted_labels, target_names=true_labels)
    return report


def write_data_to_file(data, file_path):
    """
    Write data to a file.

    Args:
    data: Data to be written to the file.
    file_path: Enter the output file path.

    Returns: None
    """
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write(data + '\n')


def main():
    """
    Pass arguments and call functions here.
    
    Args: None

    Returns: None
    """
    parser = ArgumentParser(description='This program is about generating a report from gold and predicted labels.')
    parser.add_argument('--gold', dest='g', help='Enter the gold labels file.')
    parser.add_argument('--pred', dest='p', help='Enter the predicted labels file.')
    parser.add_argument('--output', dest='o', help='Enter the output file path for all the scores.')
    args = parser.parse_args()
    gold = read_lines_from_file(args.g)
    predicted = read_lines_from_file(args.p)
    all_labels = set(predicted).union(set(gold))
    all_labels = sorted(all_labels)
    print(all_labels)
    dict_label_to_indices = {label: index for index,
                          label in enumerate(all_labels)}
    predicted_into_indexes = [dict_label_to_indices[item] for item in predicted]
    gold_into_indexes = [dict_label_to_indices[item] for item in gold]
    class_report = ''
    class_report += find_precision_recall_F1score(gold, predicted)
    if len(set(gold_into_indexes)) == 2:
        print('Micro Precision =', precision_score(gold_into_indexes, predicted_into_indexes, average='binary'))
        print('Micro Recall =', recall_score(gold_into_indexes, predicted_into_indexes, average='binary'))
        print('Micro F1 =', f1_score(gold_into_indexes, predicted_into_indexes, average='binary'))
        print('Micro Accuracy =', accuracy_score(gold_into_indexes, predicted_into_indexes))
    else:
        class_report += '\n'
        class_report += 'Micro_Precision = ' + str(precision_score(gold_into_indexes, predicted_into_indexes, average='micro')) + '\n'
        print('Micro Precision =', precision_score(gold_into_indexes, predicted_into_indexes, average='micro'))
        class_report += 'Micro_Recall = ' + str(recall_score(gold_into_indexes, predicted_into_indexes, average='micro')) + '\n'
        print('Micro Recall =', recall_score(gold_into_indexes, predicted_into_indexes, average='micro'))
        class_report += 'Micro_F1 = ' + str(f1_score(gold_into_indexes, predicted_into_indexes, average='micro')) + '\n'
        print('Micro F1 =', f1_score(gold_into_indexes, predicted_into_indexes, average='micro'))
        class_report += 'Micro_Accuracy = ' + str(accuracy_score(gold_into_indexes, predicted_into_indexes)) + '\n'
        print('Micro Accuracy =', accuracy_score(gold_into_indexes, predicted_into_indexes))
    write_data_to_file(class_report, args.o)


if __name__ == '__main__':
    main()
