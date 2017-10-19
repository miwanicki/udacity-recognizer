import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    test_lengths = test_set.get_all_Xlengths()
    for X, lengths in test_lengths.values():
        log_l_set = {}
        best_score = float("-Inf")
        best_guess = "ERROR"

        for key, model in models.items():
            try:
                temp_score = model.score(X, lengths)
                log_l_set[key] = temp_score

                if temp_score > best_score:
                    best_score = temp_score
                    best_guess = key
            except:
                log_l_set[key] = float("-Inf")

        guesses.append(best_guess)
        probabilities.append(log_l_set)

    return probabilities, guesses
