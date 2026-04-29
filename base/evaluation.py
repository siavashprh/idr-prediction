from base.data import Data


class Result:
    def __init__(self) -> None:
        self.auc = 0
        self.acc = 0
        self.f1 = 0
        self.aupr = 0
        self.recall = 0
        self.precision = 0
        self.mcc = 0
        self.max_f1 = 0
        self.tpr = None
        self.fpr = None
        self.sub_results = None

    def get_result(self):
        return {
            "AUC": self.auc,
            "ACC": self.acc,
            "F1 Score": self.f1,
            "AUPR": self.aupr,
            "Recall": self.recall,
            "Precision": self.precision,
            "MCC": self.mcc,
            "Max F1 Score": self.max_f1
        }

    def add(self, result):
        self.auc = self.auc + result.auc
        self.acc = self.acc + result.acc
        self.f1 = self.f1 + result.f1
        self.aupr = self.aupr + result.aupr
        self.precision = self.precision + result.precision
        self.recall = self.recall + result.recall
        self.mcc = self.mcc + result.mcc

    def divide(self, k):
        self.auc = self.auc / k
        self.acc = self.acc / k
        self.f1 = self.f1 / k
        self.aupr = self.aupr / k
        self.precision = self.precision / k
        self.recall = self.recall / k
        self.mcc = self.mcc / k
