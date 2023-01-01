import warnings
import numpy as np
from copy import deepcopy
from scipy.sparse import vstack
from sklearn.naive_bayes import MultinomialNB
from scipy.linalg import get_blas_funcs

warnings.filterwarnings('ignore')


# MODEL ALGORITHM PROVIDED BY https://github.com/jerry-shijieli/Text_Classification_Using_EM_And_Semisupervied_Learning

class SemiSupervised_EM_MNB():
    """
    Naive Bayes classifier for multinomial models for semi-supervised learning.
    
    Use both labeled and unlabeled data to train NB classifier, update parameters
    using unlabeled data, and all data to evaluate performance of classifier. Optimize
    classifier using log_likelihood_expectation-Maximization algorithm.
    """

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, maximum_iterations=30, tolerance=1e-6,
                 print_log_likelihood=True):
        print("Semi Supervised Learning using Expectation-Maximization")
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classifier = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior, class_prior=self.class_prior)
        self.log_likelihood = -np.inf  # log likelihood
        self.maximum_iterations = maximum_iterations  # max number of EM iterations
        self.tolerance = tolerance  # tolerance of log likelihood increment
        self.feature_log_prob_ = np.array([])  # Empirical log probability of features given a class, P(x_i|y).
        self.coef_ = np.array([])  # Mirrors feature_log_prob_ for interpreting MultinomialNB as a linear model.
        self.print_log_likelihood = print_log_likelihood  # if True, print log likelihood during EM iterations

    def fit(self, X_l, y_l, X_ul):
        """
        Initialize the parameter using labeled data only.
        Assume unlabeled class as missing values, apply EM on unlabeled data to refine classifier.
        """
        num_ul_data = X_ul.shape[0]  # number of unlabeled samples
        num_l_data = X_l.shape[0]  # number of labeled samples
        # initialization (n_docs = num_ul_data)
        classifier = deepcopy(self.classifier)  # build new copy of classifier
        classifier.fit(X_l, y_l)  # use labeled data only to initialize classifier parameters
        prev_log_likelihood = self.log_likelihood  # record log likelihood of previous EM iteration
        log_cp_word_class = classifier.feature_log_prob_  # log CP of word given class [n_classes, n_words]
        words_in_each_datarow = (X_ul > 0)  # words in each document [n_docs, n_words]
        log_cp_datarow_class = get_blas_funcs("gemm", [log_cp_word_class,
                                                       words_in_each_datarow.T.toarray()])  # log CP of doc given class [n_classes, n_docs]
        log_cp_datarow_class = log_cp_datarow_class(alpha=1.0, a=log_cp_word_class, b=words_in_each_datarow.T.toarray())
        log_prob_class = np.matrix(classifier.class_log_prior_).T  # log prob of classes [n_classes, 1]
        log_prob_class = np.repeat(log_prob_class, num_ul_data, axis=1)  # repeat for each doc [n_classes, n_docs]
        log_prob_dataRow_class = log_cp_datarow_class + log_prob_class  # joint prob of doc and class [n_classes, n_docs]
        prob_class_datarow = classifier.predict_proba(X_ul)  # weight of each class in each doc [n_docs, n_classes]
        log_likelihood_expectation = get_blas_funcs("gemm", [prob_class_datarow,
                                                             log_prob_dataRow_class])  # log_likelihood_expectation of log likelihood over all unlabeled docs
        log_likelihood_expectation = log_likelihood_expectation(alpha=1.0, a=prob_class_datarow,
                                                                b=log_prob_dataRow_class).trace()
        self.classifier = deepcopy(classifier)
        self.log_likelihood = log_likelihood_expectation
        if self.print_log_likelihood:
            print("Initial expected log likelihood = %0.3f\n" % log_likelihood_expectation)
        # Loop until log likelihood does not improve
        iter_count = 0  # count EM iteration
        print("Begin Expectation-Maximization...")
        while (self.log_likelihood - prev_log_likelihood >= self.tolerance and iter_count < self.maximum_iterations):
            # while (iter_count<self.maximum_iterations):
            iter_count += 1
            if self.print_log_likelihood:
                print("EM iteration #%d" % iter_count)  # debug
            # E-step: Estimate class membership of unlabeled documents
            y_ul = classifier.predict(X_ul)
            # M-step: Re-estimate classifier parameters
            X = vstack([X_l, X_ul])
            y = np.concatenate((y_l, y_ul), axis=0)
            classifier.fit(X, y)
            # check convergence: update log likelihood
            prob_class_datarow = classifier.predict_proba(X_ul)
            log_cp_word_class = classifier.feature_log_prob_  # log CP of word given class [n_classes, n_words]
            words_in_each_datarow = (X_ul > 0)  # words in each document
            log_cp_datarow_class = get_blas_funcs("gemm", [log_cp_word_class,
                                                           words_in_each_datarow.transpose().toarray()])  # log CP of doc given class [n_classes, n_docs]
            log_cp_datarow_class = log_cp_datarow_class(alpha=1.0, a=log_cp_word_class,
                                                        b=words_in_each_datarow.transpose().toarray())
            log_prob_class = np.matrix(classifier.class_log_prior_).T  # log prob of classes [n_classes, 1]
            log_prob_class = np.repeat(log_prob_class, num_ul_data, axis=1)  # repeat for each doc [n_classes, n_docs]
            log_prob_dataRow_class = log_cp_datarow_class + log_prob_class  # joint prob of doc and class [n_classes, n_docs]
            log_likelihood_expectation = get_blas_funcs("gemm", [prob_class_datarow,
                                                                 log_prob_dataRow_class])  # log_likelihood_expectation of log likelihood over all unlabeled docs
            log_likelihood_expectation = log_likelihood_expectation(alpha=1.0, a=prob_class_datarow,
                                                                    b=log_prob_dataRow_class).trace()
            if self.print_log_likelihood:
                print("\tExpected log likelihood = %0.3f" % log_likelihood_expectation)
            if (log_likelihood_expectation - self.log_likelihood >= self.tolerance):
                prev_log_likelihood = self.log_likelihood
                self.log_likelihood = log_likelihood_expectation
                self.classifier = deepcopy(classifier)
            else:
                break
        self.feature_log_prob_ = self.classifier.feature_log_prob_
        self.coef_ = self.classifier.coef_
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def score(self, X, y):
        return self.classifier.score(X, y)

    def __str__(self):
        return self.classifier.__str__()
