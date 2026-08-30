"""Phase 4: predict the probability that an order misses its promised window."""

from __future__ import annotations

import pandas as pd

FEATURES = [
    "distance_km",
    "stops_on_route",
    "load_kg",
    "dispatch_hour",
    "is_weekend",
    "rain_mm",
    "zone_density",
    "driver_experience_mo",
]


def train_delay_classifier(orders: pd.DataFrame, features: list[str] | None = None):
    """Gradient boosting classifier with ROC-AUC and precision/recall reporting."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split

    features = features or [f for f in FEATURES if f in orders.columns]
    X, y = orders[features], orders["is_late"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = GradientBoostingClassifier(random_state=42).fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    report = {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "report": classification_report(y_te, (proba > 0.5).astype(int), output_dict=True),
    }
    return clf, features, report


def score_orders(clf, orders: pd.DataFrame, features: list[str], threshold: float = 0.5):
    """Attach a risk score; high-risk orders are dispatched first or re-slotted."""
    out = orders.copy()
    out["late_risk"] = clf.predict_proba(out[features])[:, 1]
    out["priority"] = (out["late_risk"] >= threshold).astype(int)
    return out.sort_values("late_risk", ascending=False)


def explain(clf, X: pd.DataFrame):
    """SHAP driver explanation for operational trust in the model."""
    import shap

    explainer = shap.TreeExplainer(clf)
    return explainer.shap_values(X)
