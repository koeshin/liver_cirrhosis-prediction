# model_b_pdca_utils.py
import datetime
import numpy as np
import pandas as pd
import shap

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


# -----------------------------
# Calibration helpers
# -----------------------------
class ManualCalibratedClassifierOVR(BaseEstimator, ClassifierMixin):
    """
    Multiclass calibration by One-vs-Rest calibrators on base model probabilities.
    method:
      - 'isotonic' : IsotonicRegression (can be step-like)
      - 'sigmoid'  : LogisticRegression on probs (Platt-ish)
      - 'temperature' : grid-search temperature scaling on probs (stable for multiclass)
    NOTE: base_estimator must be pre-fitted.
    """

    def __init__(self, base_estimator, method: str = "temperature", temp_grid=None):
        self.base_estimator = base_estimator
        self.method = method
        self.temp_grid = temp_grid if temp_grid is not None else np.concatenate(
            [np.linspace(0.5, 3.0, 26), np.linspace(3.5, 10.0, 14)]
        )

        self.classes_ = None
        self.n_classes_ = 0
        self.calibrators_ = []
        self.temperature_ = 1.0

    def fit(self, X, y):
        if not hasattr(self.base_estimator, "predict_proba"):
            raise ValueError("base_estimator must support predict_proba")

        raw_probs = self.base_estimator.predict_proba(X)
        self.classes_ = getattr(self.base_estimator, "classes_", None)
        if self.classes_ is None:
            # fallback: assume 0..K-1
            self.classes_ = np.arange(raw_probs.shape[1])
        self.n_classes_ = len(self.classes_)

        y = np.asarray(y)

        if self.method == "temperature":
            # Temperature scaling on probs:
            # We approximate logits as log(probs) and scale by 1/T then renormalize.
            # Choose T minimizing NLL on calibration set.
            best_T = None
            best_nll = float("inf")

            # Avoid log(0)
            eps = 1e-12
            logp = np.log(np.clip(raw_probs, eps, 1.0))

            # Map y to indices
            class_to_idx = {c: i for i, c in enumerate(self.classes_)}
            y_idx = np.array([class_to_idx[v] for v in y], dtype=int)

            for T in self.temp_grid:
                scaled = logp / float(T)
                scaled = scaled - scaled.max(axis=1, keepdims=True)
                p = np.exp(scaled)
                p = p / p.sum(axis=1, keepdims=True)

                nll = log_loss(y_idx, p, labels=np.arange(self.n_classes_))
                if nll < best_nll:
                    best_nll = nll
                    best_T = float(T)

            self.temperature_ = 1.0 if best_T is None else best_T
            self.calibrators_ = []
            return self

        # OVR calibrators
        self.calibrators_ = []
        for i in range(self.n_classes_):
            y_binary = (y == self.classes_[i]).astype(int)
            prob_col = raw_probs[:, i].reshape(-1)

            if self.method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(prob_col, y_binary)
            elif self.method == "sigmoid":
                cal = LogisticRegression(solver="lbfgs", max_iter=1000)
                cal.fit(prob_col.reshape(-1, 1), y_binary)
            else:
                raise ValueError("method must be one of: 'temperature', 'isotonic', 'sigmoid'")

            self.calibrators_.append(cal)

        return self

    def _apply_temperature(self, probs):
        eps = 1e-12
        logp = np.log(np.clip(probs, eps, 1.0))
        scaled = logp / float(self.temperature_)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def predict_proba(self, X):
        raw_probs = self.base_estimator.predict_proba(X)

        if self.method == "temperature":
            p = self._apply_temperature(raw_probs)
            return p

        calibrated = np.zeros_like(raw_probs, dtype=float)
        for i in range(self.n_classes_):
            if self.method == "isotonic":
                calibrated[:, i] = self.calibrators_[i].transform(raw_probs[:, i])
            else:
                calibrated[:, i] = self.calibrators_[i].predict_proba(raw_probs[:, i].reshape(-1, 1))[:, 1]

        # Normalize and clip
        eps = 1e-8
        calibrated = np.clip(calibrated, eps, 1.0 - eps)
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        return calibrated

    def predict(self, X):
        p = self.predict_proba(X)
        idx = np.argmax(p, axis=1)
        return np.array([self.classes_[i] for i in idx])


# -----------------------------
# Metrics
# -----------------------------
def expected_calibration_error_multiclass(y_true_idx: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Multi-class ECE using max-prob confidence.
    y_true_idx: (n,) integer indices 0..K-1
    y_prob: (n,K)
    """
    y_true_idx = np.asarray(y_true_idx, dtype=int)
    conf = np.max(y_prob, axis=1)
    pred = np.argmax(y_prob, axis=1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true_idx)

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        in_bin = (conf > lo) & (conf <= hi)
        if not np.any(in_bin):
            continue
        acc = np.mean(pred[in_bin] == y_true_idx[in_bin])
        avg_conf = np.mean(conf[in_bin])
        ece += (np.sum(in_bin) / n) * np.abs(avg_conf - acc)

    return float(ece)


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


# -----------------------------
# PDCA Workflow
# -----------------------------
@dataclass
class DynamicSelectionConfig:
    uncertainty_threshold: float = 0.85
    score_lambda_entropy: float = 0.15
    tie_margin: float = 0.02  # if scores are too close, fallback or mark uncertain
    topk_explanations: int = 5
    calibration_method: str = "temperature"  # 'temperature' recommended
    ece_bins: int = 15


class PDCA_Workflow:
    def __init__(self, model_version: str = "Model_B_PDCA_v2", config: Optional[DynamicSelectionConfig] = None):
        self.model_version = model_version
        self.config = config if config is not None else DynamicSelectionConfig()

        # IMPORTANT: Do NOT include label(Stage) in feature groups
        self.feature_groups = {
            "Liver Biomarkers": ["Bilirubin", "Copper", "Alk_Phos", "Triglycerides", "SGOT", "Prothrombin", "Albumin"],
            "Vitals": ["Platelets", "Cholesterol"],
            "Demographics": ["Age", "Sex"],
            "Clinical Signs": ["Ascites", "Hepatomegaly", "Spiders", "Edema"],
            "Treatment": ["Drug"],
            "Time": ["N_Days"],
        }

        self.base_models: Dict[str, Any] = {}
        self.calibrated_models: Dict[str, Any] = {}
        self.explainers: Dict[str, Any] = {}

        self.classes_: Optional[np.ndarray] = None  # e.g., [0,1,2]

    # ---------- Registration ----------
    def register_candidates(self, candidates: Dict[str, Any]):
        self.base_models = candidates

    # ---------- Calibration ----------
    def calibrate_all(self, X_calib, y_calib):
        if not self.base_models:
            raise ValueError("No base models registered. Call register_candidates() first.")

        # Get class set from the first model
        first = next(iter(self.base_models.values()))
        if hasattr(first, "classes_"):
            self.classes_ = np.array(first.classes_)
        else:
            # derive from y
            self.classes_ = np.unique(y_calib)

        for name, model in self.base_models.items():
            cal = ManualCalibratedClassifierOVR(
                base_estimator=model,
                method=self.config.calibration_method
            )
            cal.fit(X_calib, y_calib)
            self.calibrated_models[name] = cal

    def evaluate_calibration_on(self, X_eval, y_eval) -> pd.DataFrame:
        if not self.calibrated_models:
            raise ValueError("No calibrated models. Call calibrate_all() first.")

        # map y_eval to indices
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[v] for v in y_eval], dtype=int)

        rows = []
        for name, cal_model in self.calibrated_models.items():
            p_cal = cal_model.predict_proba(X_eval)
            nll_cal = log_loss(y_idx, p_cal, labels=np.arange(len(self.classes_)))
            ece_cal = expected_calibration_error_multiclass(y_idx, p_cal, n_bins=self.config.ece_bins)

            # also base
            p_base = self.base_models[name].predict_proba(X_eval)
            nll_base = log_loss(y_idx, p_base, labels=np.arange(len(self.classes_)))
            ece_base = expected_calibration_error_multiclass(y_idx, p_base, n_bins=self.config.ece_bins)

            rows.append([name, "Calibrated", nll_cal, ece_cal])
            rows.append([name, "Base", nll_base, ece_base])

        df = pd.DataFrame(rows, columns=["Model", "Version", "NLL", "ECE"])
        return df

    # ---------- Dynamic Selection ----------
    def _score_model(self, probs: np.ndarray) -> Dict[str, float]:
        """
        Score for selecting the best model.
        margin = p_top1 - p_top2
        score = margin - lambda * entropy(probs)
        """
        probs = np.asarray(probs, dtype=float)
        top1 = float(np.max(probs))
        top2 = float(np.partition(probs, -2)[-2]) if probs.size >= 2 else 0.0
        margin = top1 - top2
        score = margin - self.config.score_lambda_entropy * entropy(probs)
        return {"top1": top1, "top2": top2, "margin": margin, "score": score}

    def predict_dynamic(self, X) -> List[Dict[str, Any]]:
        if not self.calibrated_models:
            raise ValueError("No calibrated models. Call calibrate_all() first.")

        results = []
        # pre-compute per-model probs
        probs_by_model = {name: cal.predict_proba(X) for name, cal in self.calibrated_models.items()}

        for i in range(len(X)):
            best = None
            second = None

            # choose best by score
            for name, P in probs_by_model.items():
                probs = P[i]
                stats = self._score_model(probs)
                cand = {
                    "model": name,
                    "probs": probs,
                    **stats
                }
                if (best is None) or (cand["score"] > best["score"]):
                    second = best
                    best = cand
                elif (second is None) or (cand["score"] > second["score"]):
                    second = cand

            assert best is not None
            pred_idx = int(np.argmax(best["probs"]))
            pred_class = self.classes_[pred_idx]

            # uncertainty logic
            status = f"Stage {int(pred_class) + 1}" if str(pred_class).isdigit() else str(pred_class)
            uncertain_reason = None

            # 1) low confidence
            if best["top1"] < self.config.uncertainty_threshold:
                status = "Uncertain"
                uncertain_reason = f"low_confidence: top1={best['top1']:.4f} < {self.config.uncertainty_threshold:.2f}"

            # 2) tie / unstable selection
            if second is not None and (best["score"] - second["score"] < self.config.tie_margin):
                status = "Uncertain"
                uncertain_reason = f"tie_or_unstable: best_score-gap={best['score'] - second['score']:.4f} < {self.config.tie_margin:.2f}"

            # If uncertain, you may choose to null out stage_prediction
            stage_prediction = int(pred_class) + 1 if status != "Uncertain" and str(pred_class).isdigit() else None
            stage_suggestion = int(pred_class) + 1 if str(pred_class).isdigit() else None

            results.append({
                "status": status,
                "stage_prediction": stage_prediction,
                "stage_suggestion": stage_suggestion,
                "selected_model": best["model"],
                "confidence": float(best["top1"]),
                "margin_score": float(best["margin"]),
                "selection_score": float(best["score"]),
                "uncertain_reason": uncertain_reason,
                "probabilities": best["probs"].tolist(),
            })

        return results

    # ---------- SHAP Explainability ----------
    def build_explainers(self, X_background: pd.DataFrame, max_background: int = 200):
        """
        Build per-model explainers.
        Uses shap.Explainer on predict_proba with background data in ORIGINAL feature space.
        (Works nicely when columns are already clean tabular features.)
        """
        if not self.base_models:
            raise ValueError("No base models registered.")

        if isinstance(X_background, pd.DataFrame):
            bg = X_background.copy()
            if len(bg) > max_background:
                bg = bg.sample(max_background, random_state=42)
        else:
            bg = X_background

        for name, model in self.base_models.items():
            # Model callable returns proba (n,K). SHAP handles multioutput.
            f = lambda data: model.predict_proba(data)
            self.explainers[name] = shap.Explainer(f, bg, feature_names=list(bg.columns) if isinstance(bg, pd.DataFrame) else None)

    def get_grouped_shap(self, model_name: str, X_one: pd.DataFrame, class_index: int) -> List[Dict[str, Any]]:
        """
        Aggregate SHAP values by clinical concept groups.
        Returns Top-k groups.
        """
        if model_name not in self.explainers:
            return [{"error": f"missing_explainer_for_{model_name}"}]

        explainer = self.explainers[model_name]

        try:
            exp = explainer(X_one)
            # exp.values: (n, features, outputs) or (n, features) depending on SHAP version
            vals = exp.values
        except Exception as e:
            return [{"error": f"shap_failed: {str(e)}"}]

        # normalize output shape to (features,)
        if isinstance(vals, np.ndarray):
            if vals.ndim == 3:
                # (n, features, outputs)
                v = vals[0, :, class_index]
            elif vals.ndim == 2:
                # (n, features)
                v = vals[0, :]
            else:
                return [{"error": f"unexpected_shap_shape: {vals.shape}"}]
        else:
            return [{"error": "unexpected_shap_values_type"}]

        feat_names = list(X_one.columns)

        grouped = {}
        for group, feats in self.feature_groups.items():
            impact = 0.0
            for f in feats:
                if f in feat_names:
                    idx = feat_names.index(f)
                    impact += float(v[idx])
            if abs(impact) > 1e-6:
                grouped[group] = impact

        # sort by abs impact
        items = sorted(grouped.items(), key=lambda x: abs(x[1]), reverse=True)
        items = items[: self.config.topk_explanations]

        out = []
        for g, val in items:
            out.append({
                "group": g,
                "value": float(val),
                "direction": "increase" if val > 0 else "decrease"
            })
        return out

    # ---------- Logging ----------
    def generate_log(self, input_row: pd.DataFrame, prediction: Dict[str, Any], explanation: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return {
            "meta": {
                "timestamp": datetime.datetime.now().isoformat(),
                "model": "Model_B_Dynamic_Ensemble",
                "selected_candidate": prediction.get("selected_model"),
                "version": self.model_version,
                "policy": f"DynamicSelection(score=margin-{self.config.score_lambda_entropy}*entropy) + "
                          f"uncertainty_threshold={self.config.uncertainty_threshold:.2f} + tie_margin={self.config.tie_margin:.2f}",
                "calibration": self.config.calibration_method,
            },
            "decision": {
                "label": prediction["status"],
                "stage_prediction": prediction.get("stage_prediction"),
                "stage_suggestion": prediction.get("stage_suggestion"),
                "confidence": prediction["confidence"],
                "margin_score": prediction["margin_score"],
                "selection_score": prediction["selection_score"],
                "reason": prediction.get("uncertain_reason"),
            },
            "explanation": explanation if explanation is not None else [],
        }

    # ---------- Validation summaries ----------
    def summarize_confidence(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        confs = np.array([p["confidence"] for p in predictions], dtype=float)
        q = np.percentile(confs, [50, 90, 95, 99]).tolist()
        uncertain_ratio = float(np.mean([p["status"] == "Uncertain" for p in predictions]))
        return {
            "p50": q[0],
            "p90": q[1],
            "p95": q[2],
            "p99": q[3],
            "uncertain_ratio": uncertain_ratio,
        }