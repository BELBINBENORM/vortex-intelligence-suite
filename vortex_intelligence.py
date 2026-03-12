import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

class VortexIntelligence:
    def __init__(self, X, y, task='classification', 
                 imbalance_threshold=3.0, skew_threshold=1.0, 
                 kurtosis_threshold=10.0, outlier_iqr_multiplier=3.0):
        self.X = X.copy().reset_index(drop=True)
        self.y = pd.Series(y).reset_index(drop=True)
        self.task = task.lower()
        self.report = None
        self.imbalance_threshold = imbalance_threshold
        self.skew_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.outlier_iqr_multiplier = outlier_iqr_multiplier

    def _generate_text_summary(self):
        if self.report is None: return
        outlier_cols = self.report[self.report['outlier_count'] > 0].shape[0]
        right_skew = self.report[self.report['skewness'] > self.skew_threshold].shape[0]
        left_skew = self.report[self.report['skewness'] < -self.skew_threshold].shape[0]
        high_kurt = self.report[self.report['kurtosis'] > self.kurtosis_threshold].shape[0]
        missing_count = self.report[self.report['null_ratio'] > 0].shape[0]

        print("\n📝 --- VORTEX INTELLIGENCE SUMMARY ---")
        if self.task == 'classification':
            ratio = self.y.value_counts().max() / self.y.value_counts().min()
            status = "🚩 High Imbalance" if ratio > self.imbalance_threshold else "✅ Balanced"
            print(f"⚖️ Balance: {status} (Ratio {ratio:.2f}:1 | Thr: {self.imbalance_threshold}:1)")
            print(f"🛡️ Predictive: Top Feature [{self.report.index[0]}] has AUC-ROC of {self.report.iloc[0]['auc_roc']:.4f} (Goal: >0.65)")

        print(f"🚩 Outliers: {f'Detected in {outlier_cols} columns' if outlier_cols > 0 else '✨ It\'s fine. No extreme outliers'} (Limit: {self.outlier_iqr_multiplier}xIQR)")
        print(f"📐 Skewness: {f'{right_skew} Right, {left_skew} Left detected' if (right_skew+left_skew) > 0 else '✨ It\'s fine. Symmetric'} (Thr: ±{self.skew_threshold})")
        print(f"🏔️ Peaks: {f'{high_kurt} columns with High Kurtosis' if high_kurt > 0 else '✨ It\'s fine. Healthy tails'} (Thr: <{self.kurtosis_threshold})")
        print(f"✨ Null Values: {'⚠️ Missing data detected' if missing_count > 0 else '✨ It\'s fine. Dataset is complete (100% density)'}")
        print(f"📏 Data Range: {self.report['min'].min():.2f} to {self.report['max'].max():.2f}")
        print(f"🎯 Feature Verdict: {self.report[self.report['vortex_action'] == '✅ STRONG SIGNAL'].shape[0]} Strong Signals.")
        print("-" * 65)

    def get_report(self):
        # 1. Feature Importance First (to get lgbm_gain)
        X_tmp = self.X.copy()
        for c in X_tmp.select_dtypes(exclude=[np.number]).columns: 
            X_tmp[c] = X_tmp[c].astype('category')
        
        model = LGBMClassifier(n_estimators=50, verbosity=-1) if self.task == 'classification' else LGBMRegressor(n_estimators=50, verbosity=-1)
        model.fit(X_tmp, self.y)
        importances = dict(zip(self.X.columns, model.feature_importances_))

        # 2. Build results row by row using a strict dictionary list
        results_list = []
        for col in self.X.columns:
            is_num = np.issubdtype(self.X[col].dtype, np.number)
            
            # Base data
            row = {
                'feature': col,
                'dtype': str(self.X[col].dtype),
                'mean': self.X[col].mean() if is_num else 0,
                'std': self.X[col].std() if is_num else 0,
                'min': self.X[col].min() if is_num else 0,
                'max': self.X[col].max() if is_num else 0,
                'skewness': self.X[col].skew() if is_num else 0,
                'kurtosis': self.X[col].kurtosis() if is_num else 0,
                'outlier_count': 0,
                'auc_roc': 0.0,
                'abs_target_corr': 0.0,
                'null_ratio': self.X[col].isnull().mean(),
                'unique_counts': self.X[col].nunique(),
                'lgbm_gain': importances.get(col, 0)
            }

            if is_num:
                # Outliers
                Q1, Q3 = self.X[col].quantile(0.25), self.X[col].quantile(0.75)
                row['outlier_count'] = ((self.X[col] < (Q1 - self.outlier_iqr_multiplier * (Q3-Q1))) | (self.X[col] > (Q3 + self.outlier_iqr_multiplier * (Q3-Q1)))).sum()
                
                # AUC-ROC
                if self.task == 'classification':
                    try:
                        score = roc_auc_score(self.y, self.X[col])
                        row['auc_roc'] = max(score, 1 - score)
                    except: row['auc_roc'] = 0.5
                
                # Correlation
                row['abs_target_corr'] = abs(self.X[col].corr(self.y))
            
            # Action Verdict
            row['vortex_action'] = "✅ STRONG SIGNAL" if row['lgbm_gain'] > 100 or row['auc_roc'] > 0.65 else "⚠️ WEAK SIGNAL"
            results_list.append(row)

        # 3. Create DataFrame and enforce order
        self.report = pd.DataFrame(results_list).set_index('feature')
        order = ['dtype', 'mean', 'std', 'min', 'max', 'skewness', 'kurtosis', 'outlier_count', 'auc_roc', 'abs_target_corr', 'null_ratio', 'unique_counts', 'lgbm_gain', 'vortex_action']
        self.report = self.report[order].sort_values('lgbm_gain', ascending=False)
        
        self._generate_text_summary()
        return self.report
