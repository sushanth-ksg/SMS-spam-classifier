import string 
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def transform_text(text):
    tokens=nltk.word_tokenize(text.lower())
    tokens=[c for c in tokens if c not in string.punctuation]
    tokens=[c for c in tokens if c not in stopwords.words('english')]
    tokens=[WordNetLemmatizer().lemmatize(t) for t in tokens]
    return ' '.join(tokens)