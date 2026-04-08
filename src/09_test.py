import pandas as pd
import numpy as np

# ============================================================================
# Configuration (fixed to match Train settings)
# ============================================================================

# Category mapping (same as 05_category.ipynb)
CATEGORY_MAPPING = {
    # DoS/DDoS family
    'DoS attacks-GoldenEye': 'DoS/DDoS',
    'DoS attacks-SlowHTTPTest': 'DoS/DDoS',
    'DDoS attacks-LOIC-HTTP': 'DoS/DDoS',
    'DDOS attack-LOIC-UDP': 'DoS/DDoS',
    
    # Brute Force (targeting authentication services)
    'FTP-BruteForce': 'Brute Force',
    
    # Web Attack (targeting web servers)
    'Brute Force -Web': 'Web Attack',
    'Brute Force -XSS': 'Web Attack',
    'SQL Injection': 'Web Attack',
    
    # Botnet
    'Bot': 'Botnet',
    
    # Benign
    'Benign': 'Benign'
}

# Final rule configuration (thresholds derived from Train)
FINAL_RULES = {
    'Brute Force': {
        'mode': 'AND',
        'alpha': 0.01,
        'conditions': {
            'Bwd Pkts/s': ('>=', 51754.3860),
            'Init Fwd Win Byts': ('>=', 26883.0000)
        }
    },
    'Botnet': {
        'mode': 'AND',
        'alpha': 0.05,
        'conditions': {
            'Flow Duration': ('<=', 20.0000),
            'Pkt Len Mean': ('<=', 0.0000)
        },
        'note': 'Failure case (Recall=0%)'
    },
    'DoS/DDoS': {
        'mode': 'AND',
        'alpha': 0.10,
        'conditions': {
            'PSH Flag Cnt': ('>=', None),  # To be filled with Step 4B result
            'Init Fwd Win Byts': ('>=', None)
        }
    },
    'Web Attack': {
        'mode': 'k-of-n',
        'k': 2,
        'alpha': 0.01,
        'conditions': {
            'Subflow Fwd Byts': ('>=', None),  # To be filled with Step 4A result
            'Fwd Header Len': ('>=', None),
            'Fwd Pkt Len Std': ('>=', None)
        }
    }
}

# Required feature list
REQUIRED_FEATURES = [
    'Bwd Pkts/s', 'Init Fwd Win Byts', 'Flow Duration', 
    'Pkt Len Mean', 'PSH Flag Cnt', 'Subflow Fwd Byts', 
    'Fwd Header Len', 'Fwd Pkt Len Std'
]


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
print("="*70)
print("Step 5: Final Validation on Test Data")
print("="*70)

# Load data
df_test = pd.read_csv(r'../data/03_test.csv')

print(f"\n[1] Data loaded successfully")
print(f"    Total samples: {len(df_test):,}")

# Check original label distribution
print(f"\n[2] Original Label distribution:")
print(df_test['Label'].value_counts().to_string())

# Apply category mapping
df_test['Category'] = df_test['Label'].map(CATEGORY_MAPPING)

# Check for unmapped labels
unmapped = df_test[df_test['Category'].isna()]['Label'].unique()
if len(unmapped) > 0:
    print(f"\n⚠️ Unmapped labels: {unmapped}")
    # Retry after stripping whitespace
    df_test['Label_clean'] = df_test['Label'].str.strip()
    df_test['Category'] = df_test['Label_clean'].map(CATEGORY_MAPPING)
    unmapped_after = df_test[df_test['Category'].isna()]['Label'].unique()
    if len(unmapped_after) > 0:
        print(f"   Still unmapped after retry: {unmapped_after}")
        print(f"   Excluding these samples: {df_test['Category'].isna().sum()}")
        df_test = df_test[df_test['Category'].notna()]
else:
    print(f"\n✓ All labels mapped successfully")

# Check category distribution
print(f"\n[3] Category distribution:")
print(df_test['Category'].value_counts().to_string())

# Verify required features exist
print(f"\n[4] Required feature check:")
missing_features = [f for f in REQUIRED_FEATURES if f not in df_test.columns]
if missing_features:
    print(f"    ❌ Missing features: {missing_features}")
    raise ValueError(f"Required features missing: {missing_features}")
else:
    print(f"    ✓ All features present")


# ============================================================================
# Evaluation Functions
# ============================================================================

def clean_feature_value(val, feature):
    """Clean feature value"""
    if pd.isna(val) or np.isinf(val):
        return np.nan
    if feature == "Init Fwd Win Byts" and val < 0:
        return np.nan
    return val


def evaluate_and_rule(df, category, conditions):
    """Evaluate AND rule"""
    attack_df = df[df['Category'] == category]
    benign_df = df[df['Category'] == 'Benign']
    
    def check_rule(row):
        for feature, (op, threshold) in conditions.items():
            val = clean_feature_value(row[feature], feature)
            if pd.isna(val):
                return False
            if op == '>=':
                if val < threshold:
                    return False
            elif op == '<=':
                if val > threshold:
                    return False
        return True
    
    attack_hits = attack_df.apply(check_rule, axis=1)
    benign_hits = benign_df.apply(check_rule, axis=1)
    
    tp = attack_hits.sum()
    fn = len(attack_df) - tp
    fp = benign_hits.sum()
    tn = len(benign_df) - fp
    
    return compute_metrics(tp, fp, fn, tn, len(attack_df), len(benign_df))


def evaluate_k_of_n_rule(df, category, conditions, k):
    """Evaluate k-of-n rule"""
    attack_df = df[df['Category'] == category]
    benign_df = df[df['Category'] == 'Benign']
    
    def check_rule(row):
        satisfied = 0
        for feature, (op, threshold) in conditions.items():
            val = clean_feature_value(row[feature], feature)
            if pd.isna(val):
                continue
            if op == '>=':
                if val >= threshold:
                    satisfied += 1
            elif op == '<=':
                if val <= threshold:
                    satisfied += 1
        return satisfied >= k
    
    attack_hits = attack_df.apply(check_rule, axis=1)
    benign_hits = benign_df.apply(check_rule, axis=1)
    
    tp = attack_hits.sum()
    fn = len(attack_df) - tp
    fp = benign_hits.sum()
    tn = len(benign_df) - fp
    
    return compute_metrics(tp, fp, fn, tn, len(attack_df), len(benign_df))


def compute_metrics(tp, fp, fn, tn, attack_total, benign_total):
    """Calculate performance metrics"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'Precision': precision, 'Recall': recall, 'F1': f1, 'FPR': fpr,
        'Attack_Total': attack_total, 'Benign_Total': benign_total
    }


# ============================================================================
# Run Final Evaluation
# ============================================================================

print("\n" + "="*70)
print("Final Validation Results (Test Data)")
print("="*70)

# ========================================
# Fill in the actual threshold values derived from Step 4 below
# ========================================

# Brute Force (Step 3 result)
FINAL_RULES['Brute Force']['conditions'] = {
    'Bwd Pkts/s': ('>=', 51754.3860),
    'Init Fwd Win Byts': ('>=', 26883.0000)
}

# Botnet (Step 3 result - failure case)
FINAL_RULES['Botnet']['conditions'] = {
    'Flow Duration': ('<=', 20.0000),
    'Pkt Len Mean': ('<=', 0.0000)
}

# DoS/DDoS (Step 4B result)
FINAL_RULES['DoS/DDoS']['conditions'] = {
    'PSH Flag Cnt': ('>=', 1.0),
    'Init Fwd Win Byts': ('>=', 14600.0)
}

# Web Attack (Step 4A result)
FINAL_RULES['Web Attack']['conditions'] = {
    'Subflow Fwd Byts': ('>=', 24782.0440),
    'Fwd Header Len': ('>=', 400.0),
    'Fwd Pkt Len Std': ('>=', 235.6157)
}


# Run evaluation
results = []

for category in ['Brute Force', 'Botnet', 'DoS/DDoS', 'Web Attack']:
    rule = FINAL_RULES[category]
    alpha = rule['alpha']
    
    print(f"\n{'='*70}")
    print(f"[{category}]")
    print(f"{'='*70}")
    
    # Print rule
    print(f"Mode: {rule['mode']}, α ≤ {alpha}")
    print("Conditions:")
    for feat, (op, thresh) in rule['conditions'].items():
        print(f"  {feat} {op} {thresh}")
    
    # Evaluate
    if rule['mode'] == 'AND':
        result = evaluate_and_rule(df_test, category, rule['conditions'])
    elif rule['mode'] == 'k-of-n':
        k = rule['k']
        print(f"  (k={k}, {k}-of-{len(rule['conditions'])})")
        result = evaluate_k_of_n_rule(df_test, category, rule['conditions'], k)
    
    # Check FPR constraint
    pass_fpr = result['FPR'] <= alpha
    
    # Print results
    print(f"\nPerformance:")
    print(f"  TP={result['TP']:,} / {result['Attack_Total']:,}")
    print(f"  FP={result['FP']:,} / {result['Benign_Total']:,}")
    print(f"  FN={result['FN']:,}, TN={result['TN']:,}")
    print(f"\n  Recall    = {result['Recall']:.4f}")
    print(f"  FPR       = {result['FPR']:.4f}  {'✓ PASS' if pass_fpr else '✗ FAIL'} (α ≤ {alpha})")
    print(f"  Precision = {result['Precision']:.4f}")
    print(f"  F1        = {result['F1']:.4f}")
    
    # Save result
    result['Category'] = category
    result['Alpha'] = alpha
    result['Pass_FPR'] = pass_fpr
    results.append(result)


# ============================================================================
# Final Summary Table
# ============================================================================

print("\n\n" + "="*70)
print("Final Summary Table (Test Data)")
print("="*70)

results_df = pd.DataFrame(results)

print(f"\n{'Category':<15} {'Recall':>10} {'FPR':>10} {'α':>8} {'Pass':>8} {'Precision':>10} {'F1':>10}")
print("-"*75)
for _, row in results_df.iterrows():
    pass_str = "✓" if row['Pass_FPR'] else "✗"
    print(f"{row['Category']:<15} {row['Recall']:>10.4f} {row['FPR']:>10.4f} {row['Alpha']:>8.2f} {pass_str:>8} {row['Precision']:>10.4f} {row['F1']:>10.4f}")


# Train vs Test Comparison
print("\n" + "="*70)
print("Train vs Test Comparison")
print("="*70)

# Train results (Step 4 final)
TRAIN_RESULTS = {
    'Brute Force': {'Recall': 0.9660, 'FPR': 0.0088, 'F1': 0.9619},
    'Botnet': {'Recall': 0.0000, 'FPR': 0.0440, 'F1': 0.0000},
    'DoS/DDoS': {'Recall': 0.5751, 'FPR': 0.0562, 'F1': 0.6958},
    'Web Attack': {'Recall': 0.2611, 'FPR': 0.0082, 'F1': 0.3767}
}

print(f"\n{'Category':<15} {'Train Recall':>14} {'Test Recall':>14} {'Train FPR':>12} {'Test FPR':>12}")
print("-"*70)
for category in ['Brute Force', 'Botnet', 'DoS/DDoS', 'Web Attack']:
    train = TRAIN_RESULTS[category]
    test = results_df[results_df['Category'] == category].iloc[0]
    
    train_recall = f"{train['Recall']:.4f}" if train['Recall'] is not None else "N/A"
    train_fpr = f"{train['FPR']:.4f}" if train['FPR'] is not None else "N/A"
    
    print(f"{category:<15} {train_recall:>14} {test['Recall']:>14.4f} {train_fpr:>12} {test['FPR']:>12.4f}")