from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_knn(X_train, y_train, n_neighbors=5, metric='euclidean'):
    """Train a K-Nearest Neighbors classifier."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    knn.fit(X_train, y_train)
    return knn

def train_logistic_regression(X_train, y_train, max_iter=1000, random_state=42):
    """Train a Logistic Regression classifier."""
    logreg = LogisticRegression(max_iter=max_iter, random_state=random_state)
    logreg.fit(X_train, y_train)
    return logreg

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest classifier."""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf
