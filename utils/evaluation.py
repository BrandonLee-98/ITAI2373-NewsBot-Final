# utils/evaluation.py
import time
from sklearn.metrics import classification_report, accuracy_score

class PerformanceEvaluator:
    def __init__(self):
        self.start_time = None

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        if self.start_time:
            latency = time.time() - self.start_time
            self.start_time = None
            return latency
        return 0

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculates accuracy and generates a detailed classification report.
        """
        report = classification_report(y_true, y_pred, output_dict=True)
        accuracy = accuracy_score(y_true, y_pred)
        return {
            "accuracy": accuracy,
            "detailed_report": report
        }
