Group Members: Miao WANG, Yixin ZHAO, Sauraj VERMA, Jiaqian MA

# Requirements
pip install spacy==2.3.5

pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

pip install torchtext==0.4.0

# Context

In this assignment we are expected to build a classifier to predict aspect-based polarities (positive, negative and neutral) of opinions in sentences, using the <aspect_category, aspect_term, sentence> triple. 

Two things are worth noting: 

* Aspect category and aspect term are already known, we only need to train a classifier to predict sentiment labels based on given aspects. 
* It is NOT a binary sentiment analysis. We have 3 categories of polarities and we have class imbalance issue: only a small part of data have ‘neutral’ labels. 

# General idea

Following the instructions, we decided to use pre-trained word embeddings and then train our classifier on the training data. 
The general workflow is as follows:

1. Prepare inputs: 

   - Preprocess the sentences (remove stop words, special characters and punctuations, lemmatization etc.)
   - Concatenate aspect category, aspect term and preprocessed sentence as the input

2. Use pre-trained embeddings to get the vector representation

3. Pass the vectors to the classifier to generate predicted polarity labels. 

   ![](https://github.com/clutchkingasiimov/ABSA-/blob/main/exercise2/workflow.png)
   
   

# Experiments 

We tried to implement both classic machine learning classifiers and deep models, with different embeddings. The results are summarized in the table below. 

| Pretrained Embedding | Classifier    | Train Accuracy | Dev Accuracy |
| -------------------- | ------------- | -------------- | ------------ |
| BERT                 | CNN           | 90.76%         | 79.21%       |
| GloVe               | CNN           | 94.49%         | 66.84%       |
| BERT                | SVM           | 95.54%       | 76.06%       |
| BERT                 | Random Forest | 95.50%     | 77.13%       |

Given the performance, we decided to go with GloVe + CNN. 



# Final Model

We tried to fine tune the hyperparameters, including the batch size, the number of filters, the dropout rate, the learning rate and the number of epochs, to further improve the mode accuracy. 

The best combination we found with grid search is: 

​	BATCH_SIZE = 5

​	N_FILERS = 200

​	DROPOUT = 0.125

​	N_EPOCHS = 12

​	LR = 1E-3

The final model structure is as follows

![](https://github.com/clutchkingasiimov/ABSA-/blob/main/exercise2/model_architecture.png)
