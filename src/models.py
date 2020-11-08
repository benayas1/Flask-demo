import torch
from benatools.torch.efficient_net import create_efn2

class ImageClassifier(torch.nn.Module):
    def __init__(self, b=0, n_outs=39, trainable_base=False):
        super(ImageClassifier, self).__init__()
        self.base = create_efn2(b=b, include_top=False)
        
        self.set_trainable(trainable_base)
        self.classifier = torch.nn.Sequential(
          torch.nn.Linear(in_features=self.get_cnn_outputs(b), out_features=512),
          torch.nn.ReLU(),
          torch.nn.LayerNorm(512),
          torch.nn.Dropout(0.25),
          torch.nn.Linear(in_features=512, out_features=n_outs),
        )

    def set_trainable(self, trainable):
        for param in self.base.parameters():
            param.requires_grad = trainable

    def get_cnn_outputs(self, b):
        outs = [1280, 1280, 1408, 1536, 1792, 2048, 2064, 2560]
        return outs[b]
        
    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x


# ML imports
from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

# from sklearn.ensemble import RandomForestClassifier
import pickle



class NLPModel(object):

    def __init__(self):
        """Simple NLP
        Attributes:
            clf: sklearn classifier model
            vectorizor: TFIDF vectorizer or similar
        """
        self.clf = MultinomialNB()
        # self.vectorizer = TfidfVectorizer(tokenizer=spacy_tok)
        self.vectorizer = TfidfVectorizer()

    def vectorizer_fit(self, X):
        """Fits a TFIDF vectorizer to the text
        """
        self.vectorizer.fit(X)

    def vectorizer_transform(self, X):
        """Transform the text data to a sparse TFIDF matrix
        """
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def train(self, X, y):
        """Trains the classifier to associate the label with the sparse matrix
        """
        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        self.clf.fit(X, y)

    def predict_proba(self, X):
        """Returns probability for the binary class '1' in a numpy array
        """
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred

    def pickle_vectorizer(self, path='chalicelib/models/TFIDFVectorizer.pkl'):
        """Saves the trained vectorizer for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer at {}".format(path))

    def pickle_clf(self, path='chalicelib/models/SentimentClassifier.pkl'):
        """Saves the trained classifier for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))
