# Initializing Classifiers
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


# Initializing Regressors
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


def create_classifiers():
    return [DummyClassifier(),
            LogisticRegression(max_iter=100000),
            SGDClassifier(random_state=69),
            RidgeClassifier(),
            GaussianNB(),
            BernoulliNB(),
            KNeighborsClassifier(n_jobs=-1),
            SVC(random_state=69),
            DecisionTreeClassifier(random_state=69),
            RandomForestClassifier(random_state=69, n_jobs=-1),
            BaggingClassifier(random_state=69, n_jobs=-1),
            AdaBoostClassifier(random_state=69),
            GradientBoostingClassifier(random_state=69),
            ExtraTreesClassifier(n_jobs=-1, random_state=69),
            HistGradientBoostingClassifier(random_state=69),
            XGBClassifier(n_jobs=-1),
            CatBoostClassifier(verbose=False, random_state=69),
            LGBMClassifier(verbose=0, force_col_wise=True, random_state=69, n_jobs=-1)]


def create_regressors():
    return [DummyRegressor(),
            LinearRegression(max_iter=100000),
            Lasso(),
            SGDRegressor(random_state=69),
            KNeighborsRegressor(),
            SVR(random_state=69),
            DecisionTreeRegressor(random_state=69),
            RandomForestRegressor(random_state=69),
            BaggingRegressor(random_state=69),
            AdaBoostRegressor(random_state=69),
            GradientBoostingRegressor(random_state=69),
            ExtraTreesRegressor(n_jobs=-1, random_state=69),
            HistGradientBoostingRegressor(random_state=69),
            XGBRegressor(n_jobs=-1),
            CatBoostRegressor(verbose=False, random_state=69),
            LGBMRegressor(verbose=0, force_col_wise=True, random_state=69)]
