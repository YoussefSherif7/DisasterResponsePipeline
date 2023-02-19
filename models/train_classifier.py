# import libraries
import sys
import nltk
import pickle
import pandas as pd
import numpy as np
import re
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_diabetes

def load_data(database_filepath):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('Responses', con=engine.connect())
    X = df.message.values
    Y = df[df.columns[4:]]
    category_names = [df.columns[4:]]
    return X, Y, category_names

def tokenize(text):
    #normalize text
    re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    #split words to tokens
    tokens = word_tokenize(text)
    
    #initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    #removing stop words
    stop_words=  stopwords.words("english")
    no_stop_words=[w for w in tokens if not w.lower().strip() in stop_words]
    
    clean_tokens = []
    for tok in no_stop_words:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
   
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2,3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        inputs: Model produced from build_model function
                X_test, Y_test from train_test_split 
            
        output: Reports the f1 score, precision and recall for each output category of the dataset 
    """
    Y_pred = model.predict(X_test)

    for i, var in enumerate(category_names):
        print('Category:', category_names[i])
        print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        

def save_model(model, model_filepath):
    #Export model as a pickle file
    pickle.dump(model, open('model.pkl', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()