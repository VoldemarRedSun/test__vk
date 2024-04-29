import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRanker, Pool
from copy import deepcopy
import numpy as np
import os
import pandas as pd
# from catboost.datasets import msrank_10k



df = pd.read_csv('intern_task.csv')
df.columns = list(range(df.shape[1]))
train_test_split_ratio = 0.75
train_df, test_df = df, df
print(train_df)
X_train = train_df.drop([0,1], axis=1).values
y_train = train_df[0].values
queries_train = train_df[1].values

X_test = test_df.drop([0, 1], axis=1).values
y_test = test_df[0].values
queries_test = test_df[1].values


num_documents = X_train.shape[0]
from collections import Counter
Counter(y_train).items()


# max_relevance = np.max(y_train)/1.0

# y_train /= max_relevance
# y_test /= max_relevance

num_queries = np.unique(queries_train).shape[0]


train = Pool(
    data=X_train,
    label=y_train,
    group_id=queries_train
)

test = Pool(
    data=X_test,
    label=y_test,
    group_id=queries_test
)


default_parameters = {
    'iterations': 2000,
    'custom_metric': ['NDCG:top=5', 'PFound', 'AverageGain:top=10'],
    'verbose': False,
    'random_seed': 0,
}

parameters = {}


def fit_model(loss_function, additional_params=None, train_pool=train, test_pool=test):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function

    if additional_params is not None:
        parameters.update(additional_params)

    model = CatBoostRanker(**parameters)
    model.fit(train_pool, eval_set=test_pool, verbose=True)

    return model


model = fit_model('RMSE', {'custom_metric': ['PrecisionAt:top=5', 'RecallAt:top=5', 'NDCG:top=5' ]})
model.get_evals_result()
model.get_best_score()
