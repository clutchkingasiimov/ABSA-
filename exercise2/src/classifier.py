import numpy
import pickle

from Utils import * 
from sklearn.ensemble import RandomForestClassifier


embedding = 'BERT'
bert_version = 'WORD'


class Classifier:
    """The Classifier"""

    def __init__(self, n_estimators=100, random_state=42):
        
        self.n_estimators = n_estimators
        self.random_state = random_state 
        


    #############################################
    def train(self, trainfile, devfile=None):
        """
        Trains the classifier model on the training set stored in file trainfile
        WARNING: DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        train_set = load_data(trainfile)
        train_set['clean_sentence_bert'] = train_set['sentence'].apply(cleanText)
        
        # get BERT embedding

        
        
        
        # generate featrues 
        bert_word_training_features = train_set['clean_sentence_bert'].apply(embeddToBERT)
        feature = [x for x in bert_word_training_features.transpose()]
        bert_word_training_features = np.asarray(feature)
        
        
        model = RandomForestClassifier(self.n_estimators, self.random_state)

        # fit the model
        self.model.fit(bert_word_training_features, 
                       train_set["polarity_2"])

        # save the fitted model
        pickle.dump(self.model, open('RF_model', 'wb'))
        


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        test_set = load_data(trainfile)
        test_set['clean_sentence_bert'] = test_set['sentence'].apply(cleanText)
    
        
        # generate featrues 
        bert_word_test_features = test_set['clean_sentence_bert'].apply(embeddToBERT)
        feature = [x for x in bert_word_test_features.transpose()]
        bert_word_test_features = np.asarray(feature)
        
        
        # load model
        loaded_model = pickle.load(open('RF_model', 'rb'))
        
        # predict
        y_pred_bert_words_rf = loaded_model.predict(bert_word_test_features)
        
        # convert numerical labels to text labels 
        label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
        slabels = [label_dict[i] for i in y_pred_bert_words_rf]
        
        return slabels

        
