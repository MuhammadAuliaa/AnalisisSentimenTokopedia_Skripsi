import numpy as np

class MultinomialNaiveBayes:
    def __init__(self):
        self.vocab_size = 0
        self.class_count = {}
        self.class_word_counts = {}
        self.class_prior = {}
        self.word_vocab = []
        self.smoothing_factor = 1

    def fit(self, X_train, y_train):
        self.vocab_size = X_train.shape[1]
        self.word_vocab = list(range(self.vocab_size))
        self.class_count = {}
        self.class_word_counts = {}
        self.class_prior = {}

        # Hitung jumlah data per kelas dan jumlah kata per kelas
        for cls in np.unique(y_train):
            X_cls = X_train[y_train == cls]
            self.class_count[cls] = X_cls.shape[0]
            self.class_word_counts[cls] = np.sum(X_cls, axis=0)

        # Hitung prior probability untuk setiap kelas
        total_samples = len(y_train)
        for cls in self.class_count:
            self.class_prior[cls] = self.class_count[cls] / total_samples

    def _predict_single(self, x):
        scores = {}
        for cls in self.class_count:
            log_prior = np.log(self.class_prior[cls])

            # Hitung log likelihood untuk setiap kelas
            log_likelihood = np.sum(
                np.log((self.class_word_counts[cls] + self.smoothing_factor) / (np.sum(self.class_word_counts[cls]) + self.smoothing_factor * self.vocab_size))
                * x
            )

            scores[cls] = log_prior + log_likelihood

        return max(scores, key=scores.get)

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            y_pred.append(self._predict_single(x))

        return np.array(y_pred)

    def predict_proba(self, X_test):
        probabilities = []
        for x in X_test:
            probs = {}
            for cls in self.class_count:
                log_prior = np.log(self.class_prior[cls])

                # Hitung log likelihood untuk setiap kelas
                log_likelihood = np.sum(
                    np.log((self.class_word_counts[cls] + self.smoothing_factor) / (np.sum(self.class_word_counts[cls]) + self.smoothing_factor * self.vocab_size))
                    * x
                )

                probs[cls] = np.exp(log_prior + log_likelihood)

            total_prob = sum(probs.values())
            prob_values = [probs[cls] / total_prob for cls in probs]
            probabilities.append(prob_values)

        return np.array(probabilities)
    