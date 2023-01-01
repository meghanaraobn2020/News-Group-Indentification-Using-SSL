import warnings
import numpy as np
from cross_validation import cross_validation
from SemiSupervised_EM_MNB import SemiSupervised_EM_MNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from Feature_extraction import FeatureExtractor
from sklearn.semi_supervised import LabelPropagation
from scipy.sparse import vstack
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def supervised_nb(X_train, y_train, split_ratio, f1scores_nb_train, f1scores_nb_test):
    print("Supervised training using Multinomial Naive Bayes")
    for split in split_ratio:
        X_l, X_u, y_l, y_u = train_test_split(X_train, y_train, train_size=split,
                                              stratify=y_train)  # Divide train data set into labeled and unlabeled data sets
        print("Training using", split * 100, "%", "labelled data")
        nb_cv_scores = list()
        nb_cv_times = list()
        batchwise_f1scores = 0
        for no_of_labelled_data in batchwise_data_splits:
            nb_clf = MultinomialNB(alpha=1e-2)
            model, cv_scores, cv_time, kfold_f1scores_avg = cross_validation(nb_clf, X_l[:no_of_labelled_data, ],
                                                                             y_l[:no_of_labelled_data])
            nb_cv_scores.append(cv_scores)
            nb_cv_times.append(cv_time)
            batchwise_f1scores += kfold_f1scores_avg
        predicted_labels = model.predict(X_test)
        f1score = f1_score(y_test, predicted_labels, average='macro')
        f1scores_nb_test.append(f1score)
        batchwise_f1scores_avg = batchwise_f1scores / len(batchwise_data_splits)
        f1scores_nb_train.append(batchwise_f1scores_avg)
        print("Training f1-score :", batchwise_f1scores_avg)
        print("Test f1-score :", f1score)


def semi_supervised_em(X_train, y_train, split_ratio, f1scores_em_train, f1scores_em_test):
    # Semi-supervised EM
    print("Semi-Supervised training using Expectation Maximization and Naive Bayes")
    # Cross validation for semisupervised EM Naive Bayes classifier 
    # using both labeled and unlabeled data set
    for split in split_ratio:
        X_l, X_u, y_l, y_u = train_test_split(X_train, y_train, train_size=split, stratify=y_train)
        print("Training using", split * 100, "%" + "labelled data")
        em_nb_cv_scores = list()
        em_nb_cv_times = list()
        batchwise_f1scores = 0
        for n_l_docs in batchwise_data_splits:
            em_nb_clf = SemiSupervised_EM_MNB(alpha=1e-2, tolerance=100,
                                              print_log_likelihood=False)  # semi supervised EM based Naive Bayes classifier
            model, cv_scores, cv_time, kfold_f1scores_avg = cross_validation(em_nb_clf, X_l[:n_l_docs, ],
                                                                             y_l[:n_l_docs], X_u)
            em_nb_cv_scores.append(cv_scores)
            em_nb_cv_times.append(cv_time)
            batchwise_f1scores += kfold_f1scores_avg
        predicted_labels = model.predict(X_test)
        f1score = f1_score(y_test, predicted_labels, average='macro')
        f1scores_em_test.append(f1score)
        batchwise_f1scores_avg = batchwise_f1scores / len(batchwise_data_splits)
        f1scores_em_train.append(batchwise_f1scores_avg)
        print("Training f1-score :", str(batchwise_f1scores_avg))
        print("Test f1-score :", f1score)


def label_propagation(X_train, y_train, split_ratio, f1scores_lp_train, f1scores_lp_test):
    print("Semi-Supervised training using Label propagation and Naive Bayes")

    # label propagation
    for split in split_ratio:
        X_l, X_u, y_l, y_u = train_test_split(X_train, y_train, train_size=split, stratify=y_train)
        print("Training using", split * 100, "%", "labelled data")
        batchwise_f1scores = 0
        # create the training dataset input
        X_train_mixed = vstack([X_l, X_u])
        # create "no label" for unlabeled data
        nolabel = [-1 for _ in range(len(y_u, ))]
        # recombine training dataset labels
        y_train_mixed = np.concatenate((y_l, nolabel), axis=0)
        # define model
        model = LabelPropagation()
        # fit model on training dataset
        model.fit(X_train_mixed.toarray(), y_train_mixed)
        # get labels for entire training dataset data
        train_labels = model.transduction_
        # define supervised learning model  
        for n_l_docs in batchwise_data_splits:
            clf = MultinomialNB(alpha=1e-2)  # semi supervised EM based Naive Bayes classifier
            model, cv_scores, cv_time, kfold_f1scores_avg = cross_validation(clf, X_train_mixed[:n_l_docs, ],
                                                                             train_labels[:n_l_docs])
            batchwise_f1scores += kfold_f1scores_avg
        batchwise_f1scores_avg = batchwise_f1scores / len(batchwise_data_splits)
        f1scores_lp_train.append(batchwise_f1scores_avg)
        print("Training f1-score :", batchwise_f1scores_avg)
        # fit supervised learning model on entire training dataset
        # clf.fit(X_train_mixed, train_labels)
        print("Predicting labels for test data")
        predicted_labels = model.predict(X_test)
        f1score = f1_score(y_test, predicted_labels, average='macro')
        f1scores_lp_test.append(f1score)
        print("Testdata f1-score :", f1score)


def plotgraph(f1scores_nb, f1scores_em, f1scores_lp, title):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(split_ratio, f1scores_nb, color='b', linewidth=2, label='Naive Bayes')
    ax.plot(split_ratio, f1scores_em, color='g', linewidth=2, label='Semisupervised EM Naive Bayes')
    ax.plot(split_ratio, f1scores_lp, color='r', linewidth=2, label='Label Propagation NB')
    ax.set_xlabel('Fraction of Labeled Documents')
    ax.set_ylabel('F1 score')
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # Downloading dataset and Feature extraction
    extractor = FeatureExtractor()
    X_train, y_train, X_test, y_test, X_vali, y_vali = extractor.extract_features()

    split_ratio = np.arange(0.2, 0.8, 0.2)
    batchwise_data_splits = np.logspace(2.3, 3.7, num=20, base=10, dtype='int')

    # global variables
    f1scores_nb_train = list()
    f1scores_nb_test = list()
    f1scores_em_train = list()
    f1scores_em_test = list()
    f1scores_lp_train = list()
    f1scores_lp_test = list()

    supervised_nb(X_train, y_train, split_ratio, f1scores_nb_train, f1scores_nb_test)
    semi_supervised_em(X_train, y_train, split_ratio, f1scores_em_train, f1scores_em_test)
    label_propagation(X_train, y_train, split_ratio, f1scores_lp_train, f1scores_lp_test)
    plotgraph(f1scores_nb_train, f1scores_em_train, f1scores_lp_train,
              "Training accuracy v/s fraction of labelled documents in Training set")
    plotgraph(f1scores_nb_test, f1scores_em_test, f1scores_lp_test, "Model's performance on test set")
