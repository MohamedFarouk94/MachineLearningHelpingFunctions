from time import time
from print_time import print_time
from copy import copy
from random import shuffle
import pandas as pd
from sklearn.metrics import roc_auc_score, cross_val_predict


def get_all_possible_dicts(output_list, main_dict, curr_i=0, total_n=None, curr_dict={}):
    total_n = total_n if total_n else len(main_dict)
    if curr_i == total_n:
        output_list.append(curr_dict)
        return

    for i, k in enumerate(main_dict):
        if i < curr_i:
            continue

        for v in main_dict[k]:
            curr_dict[k] = v
            get_all_possible_dicts(output_list, main_dict,
                                   curr_i=curr_i + 1, total_n=len(main_dict), curr_dict=copy(curr_dict))

        return


def search_model(model_to_call, all_pars, X, y, next_query=0.25,
                 method='predict_proba', cv=5, scoring=roc_auc_score):
    all_possible_pars = []
    get_all_possible_dicts(all_possible_pars, all_pars)
    shuffle(all_possible_pars)
    n_possiblities = len(all_possible_pars)
    next_query = next_query if next_query > 1 else int(n_possiblities * next_query)
    print(f'There are {n_possiblities} possibilities to check.')
    print(f'You will be asked if you want to continue after {next_query} iterations.')

    best_score = -10000
    best_pars = dict()
    iterations_since_last_high = 0
    search_df = []
    for i, pars in enumerate(all_possible_pars):
        print(f'\n({i+1}/{n_possiblities})')
        print(pars)
        model = model_to_call(**pars)

        before = time()
        y_pred = cross_val_predict(model, X, y, method=method, cv=cv)
        time_taken = print_time(before, returning=True)

        y_pred = y_pred[:, 1] if method == 'predict_proba' else y_pred
        score = scoring(y, y_pred)
        print(f'score={score}')

        if score > best_score:
            print('NEW HIGH!')
            best_score = score
            best_pars = pars
            iterations_since_last_high = 0
        else:
            iterations_since_last_high += 1

        search_df.append([i, time_taken, score])

        if i + 1 == next_query:
            print('_' * 25)
            print(f'\nModel has achived score of {best_score} until now.')
            print(f'It has been {iterations_since_last_high} iterations since the last high score.')
            print('Do you want to continue?')
            s = 'Enter 0 if you want to exit, or enter a number of iterations you want to add: '

            try:
                user_input = int(input(s))
            except ValueError:
                user_input = 0

            new_query = user_input + i + 1
            if new_query >= n_possiblities:
                print('Procedure is being continued till the end.')

            elif new_query > i + 1:
                next_query = new_query
                print(f'Procedure is being continued, You will be asked again in iteration {next_query}.')

            else:
                print('Procedure is ended.')
                break

            print('_' * 25)

    search_df = pd.DataFrame(search_df, columns=['IndexOfParametersInShuffledList', 'Time-Taken', 'Score'])
    search_df = search_df.sort_values(by='Score', ascending=False).reset_index().drop('index', axis=1)

    return best_score, best_pars, all_possible_pars, search_df
