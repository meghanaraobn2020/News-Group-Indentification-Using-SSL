import warnings
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re
from sklearn.pipeline import FeatureUnion
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
import nltk
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from scipy.sparse import hstack

warnings.filterwarnings('ignore')


class SkipGramVectorizer(CountVectorizer):
    def build_analyzer(self):
        self.tokenize = self.build_tokenizer()
        return lambda doc: self._word_skip_grams(self.tokenize(doc))

    def _word_skip_grams(self, tokens):
        skip_grams = []
        for idx in range(1, len(tokens) - 1):
            pre_word = tokens[idx - 1]
            post_word = tokens[idx + 1]
            skip_grams.append((pre_word, post_word))
        return skip_grams


class POSVectorizer(CountVectorizer):
    def build_analyzer(self):
        self.tokenize = self.build_tokenizer()
        return lambda doc: self._pos_tagging(self.tokenize(doc))

    def _pos_tagging(self, tokens):
        tag_word_pair = nltk.pos_tag(tokens)

        return tag_word_pair


class LDAVectorizerPreFit(CountVectorizer):
    lda_model = None
    lda_dictionary = None

    def __init__(self, lda_model, lda_dictionary):
        super().__init__()
        self.lda_model = lda_model
        self.lda_dictionary = lda_dictionary

    @staticmethod
    def sent_to_words(sentences):
        for sentence in sentences:
            yield (simple_preprocess(str(sentence), deacc=True))

    @staticmethod
    def get_corpus(data):
        """
        Get Bigram Model, Corpus, id2word mapping
        """

        documents = list(LDAVectorizerPreFit.sent_to_words(data))

        id2word = Dictionary(documents)  # must be list of iterable string
        id2word.filter_extremes(no_below=10, no_above=0.35)
        id2word.compactify()
        corpus = [id2word.doc2bow(text) for text in documents]
        return corpus, id2word, documents

    def build_analyzer(self):
        self.prepare_data = LDAVectorizerPreFit.get_corpus

        return lambda doc: self.lda_topics(doc)  # compose(self.tokenize, self.preprocess, self.decode)(doc),

    def lda_topics(self, tokens):
        bow = self.lda_dictionary.doc2bow(tokens.split(' '))

        lda_topics = self.lda_model.get_document_topics(bow, minimum_probability=0.0)
        topic_vec = [lda_topics[i][1] for i in range(len(lda_topics))]

        return topic_vec


class W2VVectorizer(CountVectorizer):
    def __init__(self, w2v_model):
        super().__init__()
        self.w2v_model = w2v_model

    def build_analyzer(self):
        self.tokenize = self.build_tokenizer()
        return lambda doc: self._mean_vectors(self.tokenize(doc))

    def get_mean_vector(self, words):
        # remove out-of-vocabulary words
        words = [word for word in words if word in self.w2v_model.wv]
        if len(words) >= 1:
            return np.mean(self.w2v_model.wv[words], axis=0)
        else:
            return []

    def _mean_vectors(self, tokens):
        word_vects = self.get_mean_vector(tokens)

        return word_vects


class FeatureExtractor():
    lda_model = None
    w2v_model = None
    lda_vectorizer = None
    w2v_vectorizer = None

    lda_dictionary = None

    datset = None
    transformed_dataset = None
    labels = None

    def extract_features(self):
        print("Downloading NLTK models...")
        FeatureExtractor.download_nltk_models()
        print("Downloading dataset...")
        dataset = FeatureExtractor.download_dataset()

        print("Data ", len(dataset.data))
        print("Label", len(dataset.target))
        print("Targets", len(dataset.target_names))

        print("Preparing dataset...")
        prepared_data = FeatureExtractor.prepare_data(dataset.data)
        self.dataset = prepared_data
        self.labels = dataset.target

        print("Training additional models...")
        vect_pretrained_lda, vect_pretrained_word2vec, pretrained_lda_model, pretrained_word2vec_model, lda_dictionary = FeatureExtractor.train_vectorizer_models(
            prepared_data)

        self.lda_model = pretrained_lda_model
        self.w2v_model = pretrained_word2vec_model
        self.lda_vectorizer = vect_pretrained_lda
        self.w2v_vectorizer = vect_pretrained_word2vec
        self.lda_dictionary = lda_dictionary

        print("Transformation, PCA and hstack of data...")
        tranformed_dataset = FeatureExtractor.create_feature_hstack_with_pca(prepared_data, self.lda_vectorizer,
                                                                             self.w2v_vectorizer)
        self.transformed_dataset = tranformed_dataset

        print("Splitting data...")
        X_train, y_train, X_test, y_test, X_vali, y_vali = FeatureExtractor.split_data(tranformed_dataset,
                                                                                       dataset.target, 0.8, 0.2, 0.2)

        print("Done!")
        return X_train, y_train, X_test, y_test, X_vali, y_vali

    def extract_features_without_pretraining(self):
        if self.lda_vectorizer is None or self.w2v_vectorizer is None:
            print("Run extract_features() first!")
            return None

        print("Downloading NLTK models...")
        FeatureExtractor.download_nltk_models()
        print("Downloading dataset...")
        dataset = FeatureExtractor.download_dataset()

        print("Data ", len(dataset.data))
        print("Label", len(dataset.target))
        print("Targets", len(dataset.target_names))

        print("Preparing dataset...")
        prepared_data = FeatureExtractor.prepare_data(dataset.data)
        self.dataset = prepared_data
        self.labels = dataset.target

        print("Transformation, feature dimensionality reduction and hstack of data...")
        tranformed_dataset = FeatureExtractor.create_feature_hstack_with_pca(prepared_data, self.lda_vectorizer,
                                                                             self.w2v_vectorizer)
        self.transformed_dataset = tranformed_dataset

        print("Splitting data...")
        X_train, y_train, X_test, y_test, X_vali, y_vali = FeatureExtractor.split_data(tranformed_dataset,
                                                                                       dataset.target, 0.8, 0.2, 0.2)

        print("Done!")
        return X_train, y_train, X_test, y_test, X_vali, y_vali

    @staticmethod
    def download_nltk_models():
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')
        nltk.download('wordnet')

    @staticmethod
    def download_dataset():
        twenty_train = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
        return twenty_train

    @staticmethod
    def prepare_data(dataset):
        prepared_dataset = []
        for doc in dataset:
            prepared = FeatureExtractor.prepare_document(doc)
            if len(prepared) > 20:  # ensure min data length
                prepared_dataset.append(prepared)
        return prepared_dataset

    @staticmethod
    def prepare_document(document):
        result = ''
        lemmatizer = WordNetLemmatizer()
        stopword_set = set(stopwords.words('english'))
        doc_clean = re.sub(r"\n|(\\(.?){)|}|[!$%^&#()_+|~\-={}\[\]:\";'<>?,.\/\\]|[0-9]|[@]", ' ',
                           document)  # remove punctuation
        doc_clean = re.sub('\s+', ' ', doc_clean)  # remove extra space

        wordlist = doc_clean.split(' ')

        wordlist = [lemmatizer.lemmatize(word, pos='v') for word in
                    wordlist]  # restore word to its root form (lemmatization)
        # wordlist = [poster.stem(word.lower()) for word in wordlist] # restore word to its original form (stemming)

        wordlist = [word for word in wordlist if word not in stopword_set]  # remove stopwords
        result = ' '.join(wordlist)
        return result

    @staticmethod
    def split_data(dataset, datalabels, train_ratio, test_ratio, vali_ratio):
        Sub_set_1, X_vali, Sub_set_2, y_vali = train_test_split(dataset, datalabels, test_size=test_ratio,
                                                                stratify=datalabels)

        train_test_split_ratio = test_ratio / train_ratio
        X_train, X_test, y_train, y_test = train_test_split(Sub_set_1, Sub_set_2, test_size=train_test_split_ratio,
                                                            stratify=Sub_set_2)  # 0.25 * 0.8 = 0.2

        return X_train, y_train, X_test, y_test, X_vali, y_vali

    @staticmethod
    def train_LDA_model(X_train):
        train_corpus4, train_id2word4, bigram_train4 = LDAVectorizerPreFit.get_corpus(X_train)

        pretrained_lda_model = LdaMulticore(
            corpus=train_corpus4,
            num_topics=20,
            id2word=train_id2word4,
            chunksize=100,
            workers=None,  # Max number of cores
            passes=5,  # increase later
            eval_every=1,
            per_word_topics=True)

        return pretrained_lda_model, train_id2word4

    @staticmethod
    def train_word2vect_model(X_train):

        all_words = [nltk.word_tokenize(sentence) for sentence in X_train]

        word2vec = Word2Vec(all_words, window=3, min_count=5, workers=4)

        return word2vec

    @staticmethod
    def train_vectorizer_models(X_train):
        print("Training LDA model...")

        pretrained_lda_model, lda_dictionary = FeatureExtractor.train_LDA_model(X_train)
        vect_pretrained_lda = LDAVectorizerPreFit(pretrained_lda_model, lda_dictionary)

        print("Training Word2vec model...")

        pretrained_word2vec_model = FeatureExtractor.train_word2vect_model(X_train)
        vect_pretrained_word2vec = W2VVectorizer(pretrained_word2vec_model)

        return vect_pretrained_lda, vect_pretrained_word2vec, pretrained_lda_model, pretrained_word2vec_model, lda_dictionary

    @staticmethod
    def create_feature_union(lda_vectorizer, w2v_vectorizer):
        if lda_vectorizer is None or w2v_vectorizer is None:
            print("Run extract_features() first!")
            return None
        return FeatureUnion(
            [("vect", CountVectorizer()), ("skipgram", SkipGramVectorizer()), ("postags", POSVectorizer()),
             ("lda", lda_vectorizer), ("w2v", w2v_vectorizer)])

    @staticmethod
    def create_feature_hstack_with_pca(prepared_data, lda_vectorizer, w2v_vectorizer):
        if lda_vectorizer is None or w2v_vectorizer is None:
            print("Run extract_features() first!")
            return None

        max_number_features = 2500

        print("Bag of Words")
        count_vect = CountVectorizer(max_features=max_number_features).fit_transform(prepared_data)

        print("Skip Grams")
        skip_vect = SkipGramVectorizer(max_features=max_number_features).fit_transform(prepared_data)

        print("PoS Tagging")
        pos_vect = POSVectorizer(max_features=max_number_features).fit_transform(prepared_data)

        print("LDA topics")
        lda_vectorizer.max_features = max_number_features
        lda_vect = lda_vectorizer.fit_transform(prepared_data)

        print("Word2Vec")
        w2v_vectorizer.max_features = max_number_features
        w2v_vect = w2v_vectorizer.fit_transform(prepared_data)

        return hstack([count_vect, skip_vect, pos_vect, lda_vect, w2v_vect])
