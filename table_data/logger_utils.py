import sys
import os

class LogTee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename
        self.log_file = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self.log_file = open(self.filename, 'a', encoding='utf-8')
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal
        if self.log_file:
            self.log_file.close()

    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()
