# Initializing Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Initializing Regressors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


def create_classifiers():
    return [LogisticRegression(max_iter=100000), SGDClassifier(random_state=69), GaussianNB(),
            KNeighborsClassifier(), DecisionTreeClassifier(random_state=69),
            RandomForestClassifier(random_state=69), BaggingClassifier(random_state=69),
            AdaBoostClassifier(random_state=69), GradientBoostingClassifier(random_state=69),
            XGBClassifier(), CatBoostClassifier(verbose=False, random_state=69),
            LGBMClassifier(verbose=0, force_col_wise=True, random_state=69)]


def create_regressors():
    return [LinearRegression(max_iter=100000), SGDRegressor(random_state=69),
            KNeighborsRegressor(), DecisionTreeRegressor(random_state=69),
            RandomForestRegressor(random_state=69), BaggingRegressor(random_state=69),
            AdaBoostRegressor(random_state=69), GradientBoostingRegressor(random_state=69),
            XGBRegressor(), CatBoostRegressor(verbose=False, random_state=69),
            LGBMRegressor(verbose=0, force_col_wise=True, random_state=69)]
