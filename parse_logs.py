import os, sys
from collections import OrderedDict
import csv

class LogParser:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.run_results = {}
        self.run_info = {}

    def get_final_accuracy(self, file_name):
        pid = file_name[file_name.index("pid") + len("pid"):file_name.index(".")]
        final_dev_acc_match = "Final Accuracy on dev: "
        input_dim_match = "input dimension: "
        hidden_dim_match = "hidden dimension: "
        lr_match = "learning rate: "
        drop_match = "dropout rate (0: no dropout): "
        epochs_match = "number of iterations: "
        with open(file_name) as f:
            lines = f.read()
            if final_dev_acc_match in lines:
                lines = lines.split("\n")
                for line in lines:
                    if input_dim_match in line:
                        input_dim = _get_string_after(line, input_dim_match)
                    elif hidden_dim_match in line:
                        hidden_dim = _get_string_after(line, hidden_dim_match)
                    elif lr_match in line:
                        lr = _get_string_after(line, lr_match)
                    elif drop_match in line:
                        drop = _get_string_after(line, drop_match)
                    elif epochs_match in line:
                        epochs = _get_string_after(line, epochs_match)
                    elif final_dev_acc_match in line:
                        self.run_results[pid] = _get_string_after(line, final_dev_acc_match)
                self.run_info[pid] = [input_dim, hidden_dim, lr, drop, epochs]
                return True
            else:
                return False

    def parse_logs(self):
        n_files_parsed = 0
        for log in _get_log_file_names(self.log_dir):
            if self.get_final_accuracy(log):
                n_files_parsed += 1
        print("Parsed ", n_files_parsed, " files.")
        self.run_results = OrderedDict(sorted(self.run_results.items(), key=lambda x: x[1], reverse=True))
        with open('results.csv', 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            for pid in self.run_results.keys():
                row = [pid, self.run_results[pid]]
                row.extend(self.run_info[pid])
                csv_writer.writerow([row])


def _get_log_file_names(dir_):
    return [f for f in os.listdir(dir_) if f.endswith(".log")]


def _get_string_after(line, match_string):
    return line[line.index(match_string) + len(match_string):]


if __name__ == '__main__':
    log_dir = sys.argv[1]
    log_parser = LogParser(log_dir)
    log_parser.parse_logs()