import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class classifier:

    def __init__(self) -> None:
        path_arjun = 'wine.csv'
        wine_data = pd.read_csv(path_arjun, encoding='ISO-8859-1')
        # splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(wine_data['review_description'], wine_data['variety'], test_size=0.2, random_state=42)
        # vectorizing the text data
        self.vectorizer = TfidfVectorizer(stop_words='english')
        selfX_train_vectorized = self.vectorizer.fit_transform(X_train)

        with open ('log_reg_wine.pkl','rb') as f:
            self.model = pickle.load(f)
        f.close()
        

    
    def classify(self,text) -> list:
        """ classify the wine based on review"""
        self.transformed_sentence = self.vectorizer.transform([text])
        return self.model.predict(self.transformed_sentence)
    
if __name__ == "__main__" :

    cl = classifier()
    print(cl.classify("very bad wine very very very very very very ery bad wine very very very very very very "))
    print(cl.classify("very bad wine very very very very very very ery bad wine very very very very very very "))
        