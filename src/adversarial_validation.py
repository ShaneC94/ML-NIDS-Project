import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def adversarial_validation(X_train, X_test, random_state=42):
    """
    Train a discriminator to distinguish training vs test data.
    High AUC indicates temporal drift.
    """

    # Label datasets: 0 = training, 1 = future/test
    X_train_adv = X_train.copy()
    X_test_adv = X_test.copy()

    y_train_adv = np.zeros(len(X_train_adv))
    y_test_adv = np.ones(len(X_test_adv))

    # Combine datasets
    X_adv = pd.concat([X_train_adv, X_test_adv], ignore_index=True)
    y_adv = np.concatenate([y_train_adv, y_test_adv])

    # Random split (no temporal meaning in this task)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_adv,
        y_adv,
        test_size=0.3,
        stratify=y_adv,
        random_state=random_state,
    )

    # Discriminator (simple, linear, interpretable)
    clf = LogisticRegression(
        max_iter=10000,
        class_weight="balanced"
    )
    clf.fit(X_tr, y_tr)

    # Evaluate separability
    y_prob = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)

    return auc