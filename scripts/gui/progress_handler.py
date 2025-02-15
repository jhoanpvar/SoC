# progress_handler.py

import threading
import queue

class ProgressHandler:
    def __init__(self, parent, progress_label):
        self.parent = parent
        self.progress_label = progress_label  # A ttk.Label instance
        self.progress_queue = queue.Queue()
        self.parent.after(100, self.process_queue)

    def progress_callback(self, message):
        self.progress_queue.put(message)

    def process_queue(self):
        try:
            while True:
                message = self.progress_queue.get_nowait()
                self.progress_label.config(text=message)
        except queue.Empty:
            pass
        self.parent.after(100, self.process_queue)  # Repetir cada 100 ms
