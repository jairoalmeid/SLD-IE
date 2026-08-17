"""
Cálculo formal de métricas multilabel (Precision, Recall, F1 macro/micro/weighted, Average Precision,
ROC-AUC, Matrizes Binárias, Bootstrap IC95%, Tratamento Isolado da Classe 0 e Detecção de Data Leakage).
"""

from typing import Dict, List, Any, Tuple, Optional, Set
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    hamming_loss,
    accuracy_score,
    balanced_accuracy_score
)
from src.sld.models.concept_label import (
    MULTILABEL_CLASSES,
    CONCEPT_LABEL_SHORT_NAMES,
    CONCEPT_LABEL_NAMES,
    CONCEPT_LABEL_DESCRIPTIONS
)
from src.sld.models.classification import EvaluationReport, PerClassMetrics


def compute_bootstrap_confidence_intervals(
    y_true_binary: np.ndarray,
    y_probs: np.ndarray,
    y_pred_binary: np.ndarray,
    thresholds: Dict[str, float],
    n_bootstraps: int = 2000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Estima Intervalos de Confiança de 95% (IC95%) por Bootstrap não-paramétrico
    com reamostragem no nível do parágrafo e reposição.
    """
    N = y_true_binary.shape[0]
    if N < 5 or n_bootstraps < 10:
        return {}

    rng = np.random.RandomState(random_state)
    boot_macro_f1 = []
    boot_micro_f1 = []
    boot_exact_match = []
    boot_hamming_loss = []

    boot_per_class: Dict[str, Dict[str, List[float]]] = {
        CONCEPT_LABEL_SHORT_NAMES[c]: {"precision": [], "recall": [], "f1": []}
        for c in MULTILABEL_CLASSES
    }

    for _ in range(n_bootstraps):
        idx_sample = rng.choice(N, size=N, replace=True)
        yt_s = y_true_binary[idx_sample]
        yp_s = y_pred_binary[idx_sample]

        # Métricas globais das classes ativas (1 a 5)
        try:
            m_f1 = float(f1_score(yt_s, yp_s, average="macro", zero_division=0))
            boot_macro_f1.append(m_f1)
        except Exception:
            pass

        try:
            mic_f1 = float(f1_score(yt_s, yp_s, average="micro", zero_division=0))
            boot_micro_f1.append(mic_f1)
        except Exception:
            pass

        try:
            em = float(accuracy_score(yt_s, yp_s))
            boot_exact_match.append(em)
        except Exception:
            pass

        try:
            hl = float(hamming_loss(yt_s, yp_s))
            boot_hamming_loss.append(hl)
        except Exception:
            pass

        # Métricas por classe
        for k_idx, class_id in enumerate(MULTILABEL_CLASSES):
            c_name = CONCEPT_LABEL_SHORT_NAMES[class_id]
            yt_k = yt_s[:, k_idx]
            yp_k = yp_s[:, k_idx]

            pos_k = int(np.sum(yt_k))
            pred_pos_k = int(np.sum(yp_k))

            if pos_k > 0 and pred_pos_k > 0:
                p_k = float(precision_score(yt_k, yp_k, zero_division=0))
                r_k = float(recall_score(yt_k, yp_k, zero_division=0))
                if p_k + r_k > 0:
                    f1_k = float(2 * p_k * r_k / (p_k + r_k))
                else:
                    f1_k = 0.0
                boot_per_class[c_name]["precision"].append(p_k)
                boot_per_class[c_name]["recall"].append(r_k)
                boot_per_class[c_name]["f1"].append(f1_k)
            elif pos_k > 0 and pred_pos_k == 0:
                boot_per_class[c_name]["recall"].append(0.0)
                boot_per_class[c_name]["f1"].append(0.0)
            elif pos_k == 0 and pred_pos_k > 0:
                boot_per_class[c_name]["precision"].append(0.0)
                boot_per_class[c_name]["f1"].append(0.0)

    def _calc_ci(values: List[float], min_valid: int = 50) -> Optional[Tuple[float, float]]:
        if len(values) < min_valid:
            return None
        low = float(np.percentile(values, 2.5))
        high = float(np.percentile(values, 97.5))
        return (round(low, 4), round(high, 4))

    ci_results = {
        "macro_f1_ci95": _calc_ci(boot_macro_f1),
        "micro_f1_ci95": _calc_ci(boot_micro_f1),
        "exact_match_ci95": _calc_ci(boot_exact_match),
        "hamming_loss_ci95": _calc_ci(boot_hamming_loss),
        "per_class": {}
    }

    for c_name, m_dict in boot_per_class.items():
        ci_results["per_class"][c_name] = {
            "precision_ci95": _calc_ci(m_dict["precision"]),
            "recall_ci95": _calc_ci(m_dict["recall"]),
            "f1_ci95": _calc_ci(m_dict["f1"])
        }

    return ci_results


def check_dataset_leakage(
    train_paragraph_ids: Optional[List[str]] = None,
    eval_paragraph_ids: Optional[List[str]] = None,
    train_texts: Optional[List[str]] = None,
    eval_texts: Optional[List[str]] = None,
    train_articles: Optional[List[str]] = None,
    eval_articles: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Verifica vazamento de dados (Data Leakage) entre os conjuntos de treino e avaliação.
    """
    warnings = []
    has_leakage = False

    if train_paragraph_ids and eval_paragraph_ids:
        shared_ids = set(train_paragraph_ids).intersection(set(eval_paragraph_ids))
        if shared_ids:
            has_leakage = True
            warnings.append(
                f"🚨 **Vazamento Crítico (Identificadores):** {len(shared_ids)} parágrafos aparecem tanto no treino quanto na avaliação."
            )

    if train_texts and eval_texts:
        shared_texts = set(t.strip() for t in train_texts if t).intersection(set(t.strip() for t in eval_texts if t))
        if shared_texts:
            has_leakage = True
            warnings.append(
                f"🚨 **Vazamento Crítico (Textos Duplicados):** {len(shared_texts)} textos idênticos estão presentes em ambos os conjuntos."
            )

    if train_articles and eval_articles:
        shared_arts = set(train_articles).intersection(set(eval_articles))
        if shared_arts:
            has_leakage = True
            warnings.append(
                f"⚠️ **Sobreposição de Artigos (Data Leakage de Grupo):** {len(shared_arts)} artigos possuem parágrafos divididos entre treino e teste. A divisão de dados sem vazamento (GroupSplit) deve ser utilizada para estimar generalização real."
            )

    return has_leakage, warnings


def run_consistency_checks(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    per_class_metrics: Dict[str, PerClassMetrics],
    total_paragraphs: int
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Executa verificações algébricas e metodológicas de consistência antes de finalizar o relatório.
    """
    checks = {}
    alerts = []

    # 1. Checagem de Tamanho Amostral
    if total_paragraphs < 30:
        alerts.append(
            f"⚠️ **Amostra Pequena:** O conjunto de avaliação contém apenas {total_paragraphs} parágrafos (N < 30). "
            f"As estimativas pontuais de desempenho possuem variância elevada."
        )

    # 2. Invariantes de Matriz de Confusão
    all_classes_consistent = True
    for c_name, m in per_class_metrics.items():
        total_cm = m.true_positives + m.false_positives + m.false_negatives + m.true_negatives
        if total_cm != total_paragraphs:
            all_classes_consistent = False
            alerts.append(f"❌ Inconsistência na classe {c_name}: TP+FP+FN+TN ({total_cm}) != N ({total_paragraphs}).")

        if (m.true_positives + m.false_negatives) != m.support_positive:
            all_classes_consistent = False
            alerts.append(f"❌ Inconsistência na classe {c_name}: TP+FN != Suporte Positivo.")

        if (m.true_negatives + m.false_positives) != m.support_negative:
            all_classes_consistent = False
            alerts.append(f"❌ Inconsistência na classe {c_name}: TN+FP != Suporte Negativo.")

        # Alerta de Suporte Raro
        if m.support_positive > 0 and m.support_positive < 5:
            alerts.append(
                f"⚠️ **Baixo Suporte:** A classe `{c_name}` possui apenas {m.support_positive} exemplos positivos no conjunto de avaliação."
            )
        elif m.support_positive == 0:
            alerts.append(
                f"ℹ️ **Classe Ausente:** A classe `{c_name}` não possui nenhum exemplo positivo (Suporte Positivo = 0). Métricas como Precision, Recall e F1 são indefinidas (N/A)."
            )

    checks["confusion_matrices_valid"] = all_classes_consistent

    # 3. Alerta de Métricas Perfeitas (Potencial Overfit ou Avaliação no Próprio Treino)
    active_f1s = [m.f1 for m in per_class_metrics.values() if m.f1 is not None and m.class_name != "not_relevant"]
    if active_f1s and all(abs(f - 1.0) < 1e-5 for f in active_f1s):
        alerts.append(
            "⚠️ **Alerta de Generalização:** Todas as métricas foram iguais a 1.0000 (100% de acerto). "
            "Verifique se o modelo foi avaliado nos mesmos dados utilizados para treino."
        )

    # 4. Invariante de Hamming Loss
    if total_paragraphs > 0:
        total_binary_errors = int(np.sum(y_true_binary != y_pred_binary))
        total_binary_decisions = total_paragraphs * y_true_binary.shape[1]
        expected_hl = total_binary_errors / max(1, total_binary_decisions)
        checks["hamming_loss_exact"] = round(expected_hl, 4)

    return checks, alerts


def compute_multilabel_evaluation(
    model_id: str,
    classifier_type: str,
    y_true_binary: np.ndarray,
    y_probs: np.ndarray,
    y_pred_binary: np.ndarray,
    thresholds: Dict[str, float],
    total_articles: int = 0,
    n_bootstraps: int = 2000,
    random_state: int = 42,
    train_paragraph_ids: Optional[List[str]] = None,
    eval_paragraph_ids: Optional[List[str]] = None,
    train_texts: Optional[List[str]] = None,
    eval_texts: Optional[List[str]] = None,
    train_articles: Optional[List[str]] = None,
    eval_articles: Optional[List[str]] = None,
    cv_metrics_table: Optional[List[Dict[str, Any]]] = None
) -> EvaluationReport:
    """
    Calcula relatório quantitativo completo de avaliação multilabel com:
    - Métricas detalhadas por classe (Classes 1 a 5 e Classe 0 isolada);
    - Métricas agregadas calculadas ESTRITAMENTE sobre as classes ativas 1 a 5;
    - Intervalos de Confiança de 95% via Bootstrap não-paramétrico;
    - Cardinalidade e Densidade de Rótulos;
    - Verificações de consistência e detecção de vazamento de dados.
    """
    total_paragraphs = y_true_binary.shape[0]

    # ----------------------------------------------------
    # 1. Métricas Agregadas Globais (Classes Ativas 1 a 5)
    # ----------------------------------------------------
    if total_paragraphs > 0:
        macro_f1 = float(f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_true_binary, y_pred_binary, average="micro", zero_division=0))
        weighted_f1 = float(f1_score(y_true_binary, y_pred_binary, average="weighted", zero_division=0))

        macro_p = float(precision_score(y_true_binary, y_pred_binary, average="macro", zero_division=0))
        micro_p = float(precision_score(y_true_binary, y_pred_binary, average="micro", zero_division=0))

        macro_r = float(recall_score(y_true_binary, y_pred_binary, average="macro", zero_division=0))
        micro_r = float(recall_score(y_true_binary, y_pred_binary, average="micro", zero_division=0))

        hl = float(hamming_loss(y_true_binary, y_pred_binary))
        subset_acc = float(accuracy_score(y_true_binary, y_pred_binary))

        total_pos_active = int(np.sum(y_true_binary))
        label_cardinality = total_pos_active / total_paragraphs
        label_density = total_pos_active / (total_paragraphs * 5)
    else:
        macro_f1 = micro_f1 = weighted_f1 = macro_p = micro_p = macro_r = micro_r = None
        hl = subset_acc = label_cardinality = label_density = 0.0

    # ----------------------------------------------------
    # 2. Bootstrap não-paramétrico para IC 95%
    # ----------------------------------------------------
    ci_dict = compute_bootstrap_confidence_intervals(
        y_true_binary=y_true_binary,
        y_probs=y_probs,
        y_pred_binary=y_pred_binary,
        thresholds=thresholds,
        n_bootstraps=n_bootstraps,
        random_state=random_state
    )

    per_class_metrics: Dict[str, PerClassMetrics] = {}
    confusion_matrices: Dict[str, List[List[int]]] = {}

    # ----------------------------------------------------
    # 3. Métricas Individuais por Dimensão Ativa (Classes 1 a 5)
    # ----------------------------------------------------
    for idx, class_id in enumerate(MULTILABEL_CLASSES):
        class_name = CONCEPT_LABEL_SHORT_NAMES[class_id]
        th = thresholds.get(class_name, thresholds.get(f"class_{class_id}", thresholds.get(f"C{class_id}", 0.50)))

        yt = y_true_binary[:, idx]
        yp_bin = y_pred_binary[:, idx]
        yp_score = y_probs[:, idx] if y_probs is not None and y_probs.shape[1] > idx else yp_bin.astype(float)

        cm = confusion_matrix(yt, yp_bin, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        sup_pos = tp + fn
        sup_neg = tn + fp
        prevalence = sup_pos / max(1, total_paragraphs)
        bin_acc = (tp + tn) / max(1, total_paragraphs)

        # Precision (indefinida se nenhum exemplo for previsto positivo)
        if (tp + fp) > 0:
            p_val = float(precision_score(yt, yp_bin, zero_division=0))
        else:
            p_val = None

        # Recall (indefinido se nenhum exemplo for positivo real)
        if sup_pos > 0:
            r_val = float(recall_score(yt, yp_bin, zero_division=0))
        else:
            r_val = None

        # Specificity (indefinida se nenhum exemplo for negativo real)
        if sup_neg > 0:
            spec_val = float(tn / sup_neg)
        else:
            spec_val = None

        # F1-Score
        if p_val is not None and r_val is not None and (p_val + r_val) > 0:
            f1_val = float(2 * p_val * r_val / (p_val + r_val))
        elif sup_pos == 0:
            f1_val = None
        else:
            f1_val = 0.0

        # Balanced Accuracy
        if r_val is not None and spec_val is not None:
            bal_acc = float((r_val + spec_val) / 2.0)
        else:
            bal_acc = None

        # FPR e FNR
        fpr_val = float(fp / sup_neg) if sup_neg > 0 else None
        fnr_val = float(fn / sup_pos) if sup_pos > 0 else None

        # Average Precision
        if sup_pos > 0:
            try:
                ap_val = float(average_precision_score(yt, yp_score))
            except Exception:
                ap_val = None
        else:
            ap_val = None

        # ROC-AUC (somente quando houver positivos e negativos)
        if sup_pos > 0 and sup_neg > 0:
            try:
                roc_val = float(roc_auc_score(yt, yp_score))
            except Exception:
                roc_val = None
        else:
            roc_val = None

        ci_class = ci_dict.get("per_class", {}).get(class_name, {})

        notes = []
        if sup_pos == 0:
            notes.append("Sem exemplos positivos no teste.")
        if sup_neg == 0:
            notes.append("Sem exemplos negativos no teste.")
        if tp + fp == 0:
            notes.append("Nenhuma predição positiva emitida.")

        per_class_metrics[class_name] = PerClassMetrics(
            class_name=class_name,
            threshold=th,
            support_positive=sup_pos,
            support_negative=sup_neg,
            support=sup_pos,
            prevalence=round(prevalence, 4),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=round(p_val, 4) if p_val is not None else None,
            recall=round(r_val, 4) if r_val is not None else None,
            specificity=round(spec_val, 4) if spec_val is not None else None,
            f1=round(f1_val, 4) if f1_val is not None else None,
            binary_accuracy=round(bin_acc, 4),
            balanced_accuracy=round(bal_acc, 4) if bal_acc is not None else None,
            fpr=round(fpr_val, 4) if fpr_val is not None else None,
            fnr=round(fnr_val, 4) if fnr_val is not None else None,
            average_precision=round(ap_val, 4) if ap_val is not None else None,
            roc_auc=round(roc_val, 4) if roc_val is not None else None,
            f1_ci95=ci_class.get("f1_ci95"),
            precision_ci95=ci_class.get("precision_ci95"),
            recall_ci95=ci_class.get("recall_ci95"),
            is_valid=(sup_pos > 0 and sup_neg > 0),
            note=" | ".join(notes)
        )
        confusion_matrices[class_name] = cm.tolist()

    # ----------------------------------------------------
    # 4. Tratamento Isolado da Classe 0 (Não Relevante — Derivada)
    # ----------------------------------------------------
    yt_0 = (np.sum(y_true_binary, axis=1) == 0).astype(int)
    yp_0 = (np.sum(y_pred_binary, axis=1) == 0).astype(int)

    cm0 = confusion_matrix(yt_0, yp_0, labels=[0, 1])
    tn0, fp0, fn0, tp0 = int(cm0[0, 0]), int(cm0[0, 1]), int(cm0[1, 0]), int(cm0[1, 1])

    sup_pos_0 = tp0 + fn0
    sup_neg_0 = tn0 + fp0
    prev_0 = sup_pos_0 / max(1, total_paragraphs)
    bin_acc_0 = (tp0 + tn0) / max(1, total_paragraphs)

    c0_name = CONCEPT_LABEL_SHORT_NAMES[0]
    if sup_pos_0 == 0:
        p0_val = r0_val = f1_0_val = ap0_val = roc0_val = None
        spec0_val = 1.0 if sup_neg_0 > 0 else None
        bal_acc_0 = None
        fpr0_val = 0.0 if sup_neg_0 > 0 else None
        fnr0_val = None
        c0_note = "A amostra de avaliação não contém parágrafos não relevantes; portanto, o desempenho positivo da Classe 0 não pôde ser estimado."
    else:
        p0_val = float(precision_score(yt_0, yp_0, zero_division=0)) if (tp0 + fp0) > 0 else None
        r0_val = float(recall_score(yt_0, yp_0, zero_division=0))
        spec0_val = float(tn0 / sup_neg_0) if sup_neg_0 > 0 else None
        if p0_val is not None and r0_val is not None and (p0_val + r0_val) > 0:
            f1_0_val = float(2 * p0_val * r0_val / (p0_val + r0_val))
        else:
            f1_0_val = 0.0
        bal_acc_0 = float((r0_val + (spec0_val or 0.0)) / 2.0) if spec0_val is not None else None
        fpr0_val = float(fp0 / sup_neg_0) if sup_neg_0 > 0 else None
        fnr0_val = float(fn0 / sup_pos_0) if sup_pos_0 > 0 else None
        ap0_val = float(average_precision_score(yt_0, yp_0))
        roc0_val = float(roc_auc_score(yt_0, yp_0)) if sup_pos_0 > 0 and sup_neg_0 > 0 else None
        c0_note = "Classe derivada (exclusão mútua das classes 1 a 5)."

    class_0_metrics = PerClassMetrics(
        class_name=c0_name,
        threshold=0.50,
        support_positive=sup_pos_0,
        support_negative=sup_neg_0,
        support=sup_pos_0,
        prevalence=round(prev_0, 4),
        true_positives=tp0,
        false_positives=fp0,
        false_negatives=fn0,
        true_negatives=tn0,
        precision=round(p0_val, 4) if p0_val is not None else None,
        recall=round(r0_val, 4) if r0_val is not None else None,
        specificity=round(spec0_val, 4) if spec0_val is not None else None,
        f1=round(f1_0_val, 4) if f1_0_val is not None else None,
        binary_accuracy=round(bin_acc_0, 4),
        balanced_accuracy=round(bal_acc_0, 4) if bal_acc_0 is not None else None,
        fpr=round(fpr0_val, 4) if fpr0_val is not None else None,
        fnr=round(fnr0_val, 4) if fnr0_val is not None else None,
        average_precision=round(ap0_val, 4) if ap0_val is not None else None,
        roc_auc=round(roc0_val, 4) if roc0_val is not None else None,
        is_valid=(sup_pos_0 > 0),
        note=c0_note
    )
    confusion_matrices[c0_name] = cm0.tolist()

    # ----------------------------------------------------
    # 5. Verificações de Consistência e Data Leakage
    # ----------------------------------------------------
    consistency_res, cons_alerts = run_consistency_checks(
        y_true_binary=y_true_binary,
        y_pred_binary=y_pred_binary,
        per_class_metrics=per_class_metrics,
        total_paragraphs=total_paragraphs
    )

    has_leak, leakage_alerts = check_dataset_leakage(
        train_paragraph_ids=train_paragraph_ids,
        eval_paragraph_ids=eval_paragraph_ids,
        train_texts=train_texts,
        eval_texts=eval_texts,
        train_articles=train_articles,
        eval_articles=eval_articles,
    )

    all_alerts = leakage_alerts + cons_alerts

    return EvaluationReport(
        model_id=model_id,
        classifier_type=classifier_type,
        total_articles=total_articles,
        total_paragraphs=total_paragraphs,
        active_classes_count=5,
        label_cardinality=round(label_cardinality, 4),
        label_density=round(label_density, 4),
        macro_f1=round(macro_f1, 4) if macro_f1 is not None else None,
        macro_f1_ci95=ci_dict.get("macro_f1_ci95"),
        micro_f1=round(micro_f1, 4) if micro_f1 is not None else None,
        micro_f1_ci95=ci_dict.get("micro_f1_ci95"),
        weighted_f1=round(weighted_f1, 4) if weighted_f1 is not None else None,
        macro_precision=round(macro_p, 4) if macro_p is not None else None,
        micro_precision=round(micro_p, 4) if micro_p is not None else None,
        macro_recall=round(macro_r, 4) if macro_r is not None else None,
        micro_recall=round(micro_r, 4) if micro_r is not None else None,
        hamming_loss=round(hl, 4),
        hamming_loss_ci95=ci_dict.get("hamming_loss_ci95"),
        subset_accuracy=round(subset_acc, 4),
        exact_match_ci95=ci_dict.get("exact_match_ci95"),
        per_class_metrics=per_class_metrics,
        class_0_metrics=class_0_metrics,
        confusion_matrices=confusion_matrices,
        cv_metrics_table=cv_metrics_table,
        consistency_checks=consistency_res,
        leakage_warnings=leakage_alerts,
        methodological_alerts=all_alerts,
    )


def generate_evaluation_tables(
    report: EvaluationReport
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Gera as 4 tabelas estruturadas do relatório formal:
    1. df_global: Métricas Agregadas Globais (Classes Ativas 1 a 5), Cardinalidade e Densidade de Rótulos.
    2. df_classes: Tabela Principal das 5 Classes Ativas (com IC95% do F1).
    3. df_class_0: Tabela Isolada da Classe 0 (Não Relevante — Derivada).
    4. df_cv: Tabela de Validação Cruzada (se houver folds disponíveis).
    """
    def _fmt(val: Optional[float]) -> str:
        return f"{val:.4f}" if val is not None else "N/A"

    def _fmt_ci(ci: Optional[Tuple[float, float]]) -> str:
        return f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci is not None else "N/A"

    # 1. Tabela Global das Classes Ativas 1 a 5
    global_rows = [
        {"Métrica das Classes Ativas (1–5)": "Macro-F1", "Notação": "Macro-F1", "Valor Estimado": _fmt(report.macro_f1), "IC 95% (Bootstrap)": _fmt_ci(report.macro_f1_ci95), "Interpretação Metodológica": "Média não ponderada do F1 das classes ativas, atribuindo o mesmo peso às classes frequentes e minoritárias."},
        {"Métrica das Classes Ativas (1–5)": "Micro-F1", "Notação": "Micro-F1", "Valor Estimado": _fmt(report.micro_f1), "IC 95% (Bootstrap)": _fmt_ci(report.micro_f1_ci95), "Interpretação Metodológica": "F1 global calculado a partir da soma agregada de TP, FP e FN em todas as instâncias e classes ativas."},
        {"Métrica das Classes Ativas (1–5)": "Weighted-F1", "Notação": "Weighted-F1", "Valor Estimado": _fmt(report.weighted_f1), "IC 95% (Bootstrap)": "—", "Interpretação Metodológica": "Média ponderada pelo suporte real de cada classe ativa no Gold Standard."},
        {"Métrica das Classes Ativas (1–5)": "Macro-Precision", "Notação": "Macro-P", "Valor Estimado": _fmt(report.macro_precision), "IC 95% (Bootstrap)": "—", "Interpretação Metodológica": "Média das taxas de precisão positiva entre as classes 1 a 5."},
        {"Métrica das Classes Ativas (1–5)": "Micro-Precision", "Notação": "Micro-P", "Valor Estimado": _fmt(report.micro_precision), "IC 95% (Bootstrap)": "—", "Interpretação Metodológica": "Proporção global de predições positivas que estavam corretas (TP / (TP + FP))."},
        {"Métrica das Classes Ativas (1–5)": "Macro-Recall", "Notação": "Macro-R", "Valor Estimado": _fmt(report.macro_recall), "IC 95% (Bootstrap)": "—", "Interpretação Metodológica": "Média das taxas de recuperação/sensibilidade entre as classes 1 a 5."},
        {"Métrica das Classes Ativas (1–5)": "Micro-Recall", "Notação": "Micro-R", "Valor Estimado": _fmt(report.micro_recall), "IC 95% (Bootstrap)": "—", "Interpretação Metodológica": "Proporção global de evidências conceituais reais recuperadas (TP / (TP + FN))."},
        {"Métrica das Classes Ativas (1–5)": "Subset Accuracy (Exact Match)", "Notação": "Exact Match", "Valor Estimado": _fmt(report.subset_accuracy), "IC 95% (Bootstrap)": _fmt_ci(report.exact_match_ci95), "Interpretação Metodológica": "Proporção de parágrafos em que o conjunto exato de todas as 5 classes foi previsto sem nenhum erro."},
        {"Métrica das Classes Ativas (1–5)": "Hamming Loss", "Notação": "Hamming Loss", "Valor Estimado": _fmt(report.hamming_loss), "IC 95% (Bootstrap)": _fmt_ci(report.hamming_loss_ci95), "Interpretação Metodológica": "Fração de decisões binárias individuais incorretas (menor é melhor)."},
        {"Métrica das Classes Ativas (1–5)": "Cardinalidade de Rótulos", "Notação": "LC", "Valor Estimado": f"{report.label_cardinality:.4f}", "IC 95% (Bootstrap)": "—", "Interpretação Metodológica": "Número médio de dimensões ativas atribuídas por parágrafo (Total Rótulos Positivos / N)."},
        {"Métrica das Classes Ativas (1–5)": "Densidade de Rótulos", "Notação": "LD", "Valor Estimado": f"{report.label_density:.4f}", "IC 95% (Bootstrap)": "—", "Interpretação Metodológica": "Proporção de pares (parágrafo, classe) ativos (Total Rótulos Positivos / (N × 5))."}
    ]
    df_global = pd.DataFrame(global_rows)

    # 2. Tabela Principal das 5 Classes Ativas
    class_rows = []
    for c_id in MULTILABEL_CLASSES:
        c_short = CONCEPT_LABEL_SHORT_NAMES[c_id]
        m = report.per_class_metrics.get(c_short)
        if m:
            class_rows.append({
                "Classe": f"Classe {c_id}",
                "Dimensão": CONCEPT_LABEL_NAMES[c_id],
                "Limiar": f"{m.threshold:.2f}",
                "Suporte positivo": m.support_positive,
                "Prevalência": f"{m.prevalence:.2%}",
                "TP": m.true_positives,
                "FP": m.false_positives,
                "FN": m.false_negatives,
                "TN": m.true_negatives,
                "Precisão": _fmt(m.precision),
                "Recall": _fmt(m.recall),
                "Especificidade": _fmt(m.specificity),
                "F1": _fmt(m.f1),
                "IC95% do F1": _fmt_ci(m.f1_ci95),
                "AP": _fmt(m.average_precision),
                "ROC-AUC": _fmt(m.roc_auc)
            })
    df_classes = pd.DataFrame(class_rows)

    # 3. Tabela Isolada da Classe 0
    c0_m = report.class_0_metrics
    if c0_m:
        df_class_0 = pd.DataFrame([{
            "Classe": "Classe 0",
            "Dimensão": "0 — Não relevante (Derivada)",
            "Suporte positivo": c0_m.support_positive,
            "Suporte negativo": c0_m.support_negative,
            "Prevalência": f"{c0_m.prevalence:.2%}",
            "TP": c0_m.true_positives,
            "FP": c0_m.false_positives,
            "FN": c0_m.false_negatives,
            "TN": c0_m.true_negatives,
            "Precisão": _fmt(c0_m.precision),
            "Recall": _fmt(c0_m.recall),
            "Especificidade": _fmt(c0_m.specificity),
            "F1": _fmt(c0_m.f1),
            "AP": _fmt(c0_m.average_precision),
            "ROC-AUC": _fmt(c0_m.roc_auc),
            "Nota Metodológica": c0_m.note or "Derivada por exclusão mútua."
        }])
    else:
        df_class_0 = pd.DataFrame()

    # 4. Tabela de Validação Cruzada (se houver)
    if report.cv_metrics_table:
        df_cv = pd.DataFrame(report.cv_metrics_table)
    else:
        df_cv = None

    return df_global, df_classes, df_class_0, df_cv


def generate_evaluation_markdown(
    report: EvaluationReport
) -> str:
    """Gera relatório metodológico completo e formal de avaliação em Markdown estruturado."""
    df_g, df_c, df_c0, df_cv = generate_evaluation_tables(report)

    md = (
        f"# Relatório de Avaliação Supervisionada Multilabel — SLD\n\n"
        f"- **Identificador do Modelo:** `{report.model_id}`\n"
        f"- **Arquitetura / Algoritmo:** `{report.classifier_type}`\n"
        f"- **Data/Hora da Avaliação:** `{report.evaluation_date}`\n"
        f"- **Total de Parágrafos Avaliados ($N$):** `{report.total_paragraphs:,}`\n".replace(",", ".") +
        f"- **Total de Documentos Representados:** `{report.total_articles}`\n"
        f"- **Cardinalidade de Rótulos ($LC$):** `{report.label_cardinality:.4f}` (média de classes ativas por parágrafo)\n"
        f"- **Densidade de Rótulos ($LD$):** `{report.label_density:.4f}`\n\n"
    )

    # Alertas Metodológicos e de Data Leakage
    if report.methodological_alerts:
        md += "## ⚠️ Alertas Metodológicos e de Integridade\n\n"
        for alert in report.methodological_alerts:
            md += f"- {alert}\n"
        md += "\n---\n\n"

    # 1. Métricas Globais das Classes Ativas (1 a 5)
    md += (
        f"## 1. Métricas Agregadas Globais (Classes Ativas 1 a 5)\n\n"
        f"> **Definição Metodológica do Macro-F1:** Média não ponderada do F1 das classes ativas, "
        f"atribuindo o mesmo peso às classes frequentes e minoritárias.\n\n"
        f"{df_g.to_markdown(index=False)}\n\n"
        f"---\n\n"
    )

    # 2. Desempenho Detalhado por Classe Ativa
    md += (
        f"## 2. Desempenho Detalhado por Dimensão Conceitual (Classes Ativas 1 a 5)\n\n"
        f"{df_c.to_markdown(index=False)}\n\n"
        f"---\n\n"
    )

    # 3. Desempenho Isolado da Classe 0
    md += (
        f"## 3. Desempenho e Diagnóstico da Classe 0 (Não Relevante — Derivada e Excludente)\n\n"
        f"> **Regra de Derivação:** A Classe 0 não é treinada diretamente como uma classe independente. "
        r"Ela é atribuída quando $\sum_{k=1}^5 y_k = 0$ (nenhuma das classes ativas 1 a 5 está presente)."
        "\n\n"
    )
    if not df_c0.empty:
        md += f"{df_c0.to_markdown(index=False)}\n\n"
    if report.class_0_metrics and report.class_0_metrics.support_positive == 0:
        md += f"> **Nota Metodológica:** {report.class_0_metrics.note}\n\n"
    md += "---\n\n"

    # 4. Tabela de Validação Cruzada (se houver)
    if df_cv is not None and not df_cv.empty:
        md += (
            f"## 4. Variação entre Folds da Validação Cruzada (K-Fold Cross-Validation)\n\n"
            f"{df_cv.to_markdown(index=False)}\n\n"
            f"---\n\n"
        )

    # 5. Matrizes de Confusão Binárias
    md += "## 5. Matrizes de Confusão Binárias 2×2 por Classe\n\n"
    for c_id in MULTILABEL_CLASSES:
        c_short = CONCEPT_LABEL_SHORT_NAMES[c_id]
        m = report.per_class_metrics.get(c_short)
        if m:
            md += (
                f"### Classe {c_id} — {CONCEPT_LABEL_NAMES[c_id]}\n"
                f"- **Verdadeiro Positivo (TP):** `{m.true_positives}` | **Falso Positivo (FP):** `{m.false_positives}`\n"
                f"- **Falso Negativo (FN):** `{m.false_negatives}` | **Verdadeiro Negativo (TN):** `{m.true_negatives}`\n"
                f"- **Acurácia Binária:** `{m.binary_accuracy:.2%}` | **Taxa FP (FPR):** `{m.fpr if m.fpr is not None else 'N/A'}` | **Taxa FN (FNR):** `{m.fnr if m.fnr is not None else 'N/A'}`\n\n"
            )

    return md


def compute_pr_curves_data(
    y_true_binary: np.ndarray,
    y_probs: np.ndarray
) -> Dict[str, Dict[str, List[float]]]:
    """Gera pontos (precision, recall) para plotagem de curvas Precision-Recall com Plotly."""
    pr_data = {}
    for idx, class_id in enumerate(MULTILABEL_CLASSES):
        class_name = CONCEPT_LABEL_SHORT_NAMES[class_id]
        yt = y_true_binary[:, idx]
        yp = y_probs[:, idx] if y_probs is not None and y_probs.shape[1] > idx else yt.astype(float)

        if np.sum(yt) > 0:
            precision_pts, recall_pts, _ = precision_recall_curve(yt, yp)
            pr_data[class_name] = {
                "precision": precision_pts.tolist(),
                "recall": recall_pts.tolist(),
            }
        else:
            pr_data[class_name] = {"precision": [0.0], "recall": [0.0]}

    return pr_data


