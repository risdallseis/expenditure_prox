import unidecode
import pandas as pd
import contractions
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Contains code perform preprocessing on reviews
# Steps included:


def __remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    
    new_list = []
    if text != None:
        for item in text:
            newtext = unidecode.unidecode(item)
            new_list.append(newtext)
        
    return new_list


def __expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    new_list = []
    if text != None:
        for item in text:
            newtext = contractions.fix(item)
            new_list.append(newtext)
        
    return new_list


def __lower_text(text):
    """Lowers cases all text
    """
    new_list = []
    if text != None:
        for item in text:
            newtext = item.lower()
            new_list.append(newtext)
        
    return new_list


def __remove_spec_chars(text):
    """Removes special characters from text
    """
    new_list = []
    if text != None:
        for item in text:
            new_list.append(re.sub('[^A-Za-z0-9]+', ' ', item))
    
    return new_list


def __remove_html(text):
    """Removes any html in text
    """
    new_list = []
    if text != None:
        for item in text:
            new_list.append(re.sub(r"'<.*?>'", "", item))
        
    return new_list


def __tokenise_reviews(text):
    """Splits up sentences to words
    """
    new_list = []
    if text != None:
        for item in text:
            new_list.append(item.split())
        
    return new_list


def __lemmatise_words(text):
    """Lemmatises tokenised words
    """
    lemmatizer = WordNetLemmatizer()
    new_list = []
    if text != None:
        for item in text:
            temp_list = []
            for word in item:
                temp_list.append(lemmatizer.lemmatize(word))
            new_list.append(temp_list)
    
    return new_list


def __remove_stopwords(text):
    """
    """
    if text != None:
        swords = stopwords.words('english')
        stopwords2remove = ['again','no','not','should']
        stopswords= [word for word in swords if word not in stopwords2remove]
        
        new_list = []
        for item in text:
            temp_list = [word for word in item if word not in swords]
            new_list.append(temp_list)
        
    return new_list


def clean_cols(df):
    """Removes unwanted text from no_reviews and casts to int
    """
    # no_reviews
    df['no_reviews'] =  [(re.findall(r'\d+', x[0]))[0] 
                         if isinstance(x, list) 
                         else 0 
                         for x in df['no_reviews']]
    df['no_reviews'] = df['no_reviews'].astype('int')
    # recommendation_percent
    df['recommendation_percent'] =  [(re.findall(r'\d+', x[0]))[0]
                                     if isinstance(x, list) 
                                     else 0 
                                     for x in df['recommendation_percent']]
    df['recommendation_percent'] = df['recommendation_percent'].astype('int')
    # summary_star_rating
    df['summary_star_rating'] =  [x[0] 
                                  if isinstance(x, list) 
                                  else 0 
                                  for x in df['summary_star_rating']]
    df['summary_star_rating'] = df['summary_star_rating'].astype('float')
    
    return df

def fill_empty_lists(df):
    """Fills empty lists with 0
    """
    df['no_reviews'] = df['no_reviews'].apply(lambda y: None if y[0]=='' else y).fillna(0)
    df['recommendation_percent'] = df['recommendation_percent'].apply(lambda y: None if y[0]=='' else y).fillna(0)
    df['review_date'] = df['review_date'].apply(lambda y: None if y[0]=='' else y).fillna(0)
    df['review_rating'] = df['review_rating'].apply(lambda y: None if y[0]=='' else y).fillna(0)
    df['review_text'] = df['review_text'].apply(lambda y: None if y[0]=='' else y).fillna(0)
    df['review_title'] = df['review_title'].apply(lambda y: None if y[0]=='' else y).fillna(0)
    df['summary_star_rating'] = df['summary_star_rating'].apply(lambda y: None if y[0]=='' else y).fillna(0)
    
    return df

def preprocess_reviews(df):
    """Function to apply all preprocessing steps to reviews dataframe
    """
    nltk.download('wordnet')
    nltk.download('stopwords')
    # Drop duplicate rows
    df = df.drop_duplicates()
    # Replace any accented characters with non-accented letter
    df['text_preproc'] = df['review_text'].apply(lambda x: __remove_accented_chars(x))
    # Lowercase all text
    df['text_preproc'] = df['text_preproc'].apply(lambda x: __lower_text(x))
    # Remove any html in text
    df['text_preproc'] = df['text_preproc'].apply(lambda x: __remove_html(x))
    # Remove any special characters, i.e !
    df['text_preproc'] = df['text_preproc'].apply(lambda x: __remove_spec_chars(x))
    # Expand contractions, i.e don't > do not
    df['text_preproc'] = df['text_preproc'].apply(lambda x: __expand_contractions(x))
    # Tokenise text - splits sentences to list of words
    df['text_preproc'] = df['text_preproc'].apply(lambda x: __tokenise_reviews(x))
    ## Preproc done
    # Remove stopwords
    df['text_preproc2'] = df['text_preproc'].apply(lambda x: __remove_stopwords(x))
    # Lemmatise words
    df['text_preproc2'] = df['text_preproc2'].apply(lambda x: __lemmatise_words(x))
    
    return df
    