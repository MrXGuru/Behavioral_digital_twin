"""Q1 Research Paper Experiments: Ablation and Statistical Significance.

This script runs a rigorous evaluation suite for the Behavioral Digital Twin:
1. Ablation Study: Tests the LSTM model under varying sequence lengths (K) and
   isolates the impact of context vs. sequence features.
2. Statistical Significance: Uses Time-Series Cross-Validation and McNemar's Test
   to prove the performance difference between models is statistically significant.
   Also includes a Markov Chain baseline.

NOTE: For actual Q1 journal submission, the synthetic data generator must be
replaced with real data collected via the `integrations.json` sync mechanism.
"""

from __future__ import annotations

import warnings
import numpy as np
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar

from data.schema import Domain, options
from seed_db import generate_user_data
from features.feature_pipeline import FeaturePipeline
from models.baseline import BaselineModel
from models.sequence import SequenceModel
from models.markov import MarkovModel
from models.evaluate import _score_model, time_based_split

warnings.filterwarnings("ignore")

USER_ID = "q1_evaluation_user"
N_RECORDS = 2000
DOMAIN = "focus"  # We'll evaluate on the focus domain
CV_FOLDS = 5

def run_ablation_study(records):
    print(f"\\n{'='*40}")
    print("EXPERIMENT 1: ABLATION STUDY")
    print(f"{'='*40}\\n")
    
    dom = Domain(DOMAIN)
    dom_records = [r for r in records if r.domain == dom.value]
    train, val = time_based_split(dom_records, 0.2)
    opts = list(options(dom))

    # Test varying K
    k_values = [1, 3, 5]
    print("--- 1a. Impact of Sequence Length (K) ---")
    print("| K | Accuracy | Macro-F1 | Brier Score |")
    print("|---|---|---|---|")
    for k in k_values:
        pipe = FeaturePipeline(dom, k=k).fit(train)
        fm_train = pipe.transform(train)
        fm_val = pipe.transform(val)
        
        seq_model = SequenceModel(opts).fit(fm_train.X, fm_train.seq, fm_train.y)
        metrics = _score_model(seq_model, fm_val, opts)
        print(f"| K={k} | {metrics.accuracy:.4f} | {metrics.macro_f1:.4f} | {metrics.brier:.4f} |")
        
    print("\\n--- 1b. Isolation of Context vs. Sequence (K=5) ---")
    # Base pipeline with K=5
    pipe = FeaturePipeline(dom, k=5).fit(train)
    fm_train = pipe.transform(train)
    fm_val = pipe.transform(val)
    
    # Full Model
    full_model = SequenceModel(opts).fit(fm_train.X, fm_train.seq, fm_train.y)
    metrics_full = _score_model(full_model, fm_val, opts)
    
    # Context Only (Zero out sequence tensor)
    zero_seq_train = np.zeros_like(fm_train.seq)
    zero_seq_val = np.zeros_like(fm_val.seq)
    ctx_model = SequenceModel(opts).fit(fm_train.X, zero_seq_train, fm_train.y)
    # Correct validation scoring using the zeroed sequence
    fm_val_ctx_only = pipe.transform(val)
    fm_val_ctx_only.seq = zero_seq_val
    metrics_ctx = _score_model(ctx_model, fm_val_ctx_only, opts)

    # Sequence Only (Zero out context matrix, leaving only temporal/rolling/embedding)
    # Actually, we zero out everything except the sequence tensor.
    zero_X_train = np.zeros_like(fm_train.X)
    zero_X_val = np.zeros_like(fm_val.X)
    seq_only_model = SequenceModel(opts).fit(zero_X_train, fm_train.seq, fm_train.y)
    fm_val_seq_only = pipe.transform(val)
    fm_val_seq_only.X = zero_X_val
    metrics_seq_only = _score_model(seq_only_model, fm_val_seq_only, opts)

    print("| Condition | Accuracy | Macro-F1 | Brier Score |")
    print("|---|---|---|---|")
    print(f"| Full Model | {metrics_full.accuracy:.4f} | {metrics_full.macro_f1:.4f} | {metrics_full.brier:.4f} |")
    print(f"| Sequence Only | {metrics_seq_only.accuracy:.4f} | {metrics_seq_only.macro_f1:.4f} | {metrics_seq_only.brier:.4f} |")
    print(f"| Context Only | {metrics_ctx.accuracy:.4f} | {metrics_ctx.macro_f1:.4f} | {metrics_ctx.brier:.4f} |")

def time_series_cv(records, n_splits=5):
    """Yields (train, val) splits for time-series cross validation."""
    dom_records = sorted([r for r in records if r.domain == DOMAIN], key=lambda r: r.timestamp)
    n = len(dom_records)
    fold_size = n // (n_splits + 1)
    
    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        val_end = (i + 1) * fold_size
        yield dom_records[:train_end], dom_records[train_end:val_end]

def run_statistical_tests(records):
    print(f"\\n{'='*40}")
    print("EXPERIMENT 2: STATISTICAL SIGNIFICANCE")
    print(f"{'='*40}\\n")
    
    opts = list(options(Domain(DOMAIN)))
    
    # Store predictions across all folds
    y_true_all = []
    y_pred_baseline = []
    y_pred_markov = []
    y_pred_lstm = []
    
    brier_baseline = []
    brier_lstm = []
    
    for fold, (train, val) in enumerate(time_series_cv(records, n_splits=CV_FOLDS)):
        pipe = FeaturePipeline(Domain(DOMAIN), k=5).fit(train)
        fm_train = pipe.transform(train)
        fm_val = pipe.transform(val)
        
        # Train Models
        baseline = BaselineModel(opts).fit(fm_train.X, fm_train.seq, fm_train.y)
        markov = MarkovModel(opts).fit(fm_train.X, fm_train.seq, fm_train.y)
        lstm = SequenceModel(opts).fit(fm_train.X, fm_train.seq, fm_train.y)
        
        # Get predictions for this fold
        for i in range(len(fm_val.y)):
            x = fm_val.X[i:i + 1]
            seq = fm_val.seq[i:i + 1]
            
            p_base = baseline.predict_proba(x, seq)
            p_markov = markov.predict_proba(x, seq)
            p_lstm = lstm.predict_proba(x, seq)
            
            y_true_all.append(fm_val.y[i])
            y_pred_baseline.append(max(p_base, key=p_base.get))
            y_pred_markov.append(max(p_markov, key=p_markov.get))
            y_pred_lstm.append(max(p_lstm, key=p_lstm.get))
        
        # Collect fold-level metrics for Paired t-test
        b_metrics = _score_model(baseline, fm_val, opts)
        l_metrics = _score_model(lstm, fm_val, opts)
        brier_baseline.append(b_metrics.brier)
        brier_lstm.append(l_metrics.brier)
        
    # Calculate McNemar's Test (LSTM vs Baseline)
    # contingency table: [[both correct, lstm correct/base wrong], [lstm wrong/base correct, both wrong]]
    b_vs_l = [[0, 0], [0, 0]]
    m_vs_l = [[0, 0], [0, 0]]
    
    total_acc_lstm = 0
    total_acc_base = 0
    total_acc_markov = 0
    
    for yt, yb, ym, yl in zip(y_true_all, y_pred_baseline, y_pred_markov, y_pred_lstm):
        l_corr = yt == yl
        b_corr = yt == yb
        m_corr = yt == ym
        
        if l_corr: total_acc_lstm += 1
        if b_corr: total_acc_base += 1
        if m_corr: total_acc_markov += 1
        
        b_vs_l[1 - int(l_corr)][1 - int(b_corr)] += 1
        m_vs_l[1 - int(l_corr)][1 - int(m_corr)] += 1

    n = len(y_true_all)
    print(f"Total Samples Across {CV_FOLDS} Folds: {n}")
    print(f"Overall Accuracy - Baseline: {total_acc_base/n:.4f}")
    print(f"Overall Accuracy - Markov:   {total_acc_markov/n:.4f}")
    print(f"Overall Accuracy - LSTM:     {total_acc_lstm/n:.4f}\\n")

    res_base = mcnemar(b_vs_l, exact=False, correction=True)
    res_markov = mcnemar(m_vs_l, exact=False, correction=True)
    
    print("--- 2a. McNemar's Test (Accuracy Significance) ---")
    print(f"LSTM vs. Baseline: statistic={res_base.statistic:.4f}, p-value={res_base.pvalue:.4e}")
    if res_base.pvalue < 0.05:
        print("  -> Result: The difference in accuracy is STATISTICALLY SIGNIFICANT.")
    else:
        print("  -> Result: The difference is NOT statistically significant.")
        
    print(f"LSTM vs. Markov:   statistic={res_markov.statistic:.4f}, p-value={res_markov.pvalue:.4e}")
    if res_markov.pvalue < 0.05:
        print("  -> Result: The difference in accuracy is STATISTICALLY SIGNIFICANT.")
    else:
        print("  -> Result: The difference is NOT statistically significant.")
        
    print("\\n--- 2b. Paired t-test (Calibration / Brier Score) ---")
    t_stat, p_val_t = stats.ttest_rel(brier_baseline, brier_lstm)
    print(f"LSTM vs. Baseline Brier Scores across folds:")
    print(f"t-statistic={t_stat:.4f}, p-value={p_val_t:.4e}")
    if p_val_t < 0.05:
        print("  -> Result: The improvement in calibration is STATISTICALLY SIGNIFICANT.")
    else:
        print("  -> Result: The difference in calibration is NOT statistically significant.")

if __name__ == "__main__":
    print("Generating simulated dataset (Note: replace with real data for final publication)...")
    records = generate_user_data(USER_ID, N_RECORDS, "predictable")
    
    run_ablation_study(records)
    run_statistical_tests(records)
