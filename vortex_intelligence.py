import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

class VortexIntelligence:
    """
    Vortex Intelligence Suite - High Performance Data Audit
    Corrected: Column alignment for AUC and Correlation.
    """
    def __init__(self, X, y, task='classification', 
                 imbalance_threshold=3.0, 
                 skew_threshold=1.0, 
                 kurtosis_threshold=10.0, 
                 outlier_iqr_multiplier=3.0):
        
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
            counts = self.y.value_counts()
            ratio = counts.max() / counts.min()
            status = "High Imbalance" if ratio > self.imbalance_threshold else "Well-balanced"
            print(f"⚖️ Balance: {status} (Ratio {ratio:.2f}:1).")
            
            top_f = self.report.index[0]
            top_a = self.report.iloc[0]['auc_roc']
            print(f"🛡️ Predictive: Top Feature [{top_f}] has AUC-ROC of {top_a:.4f}")

        if outlier_cols > 0:
            print(f"🚩 Outliers: Detected in {outlier_cols} columns ({self.outlier_iqr_multiplier}xIQR).")
        else:
            print("✨ Outliers: No extreme outliers found.")

        if (right_skew + left_skew) > 0:
            print(f"📐 Skewness: {right_skew} Right, {left_skew} Left detected (±{self.skew_threshold}).")
        else:
            print(f"✨ Skewness: All features are symmetric (within ±{self.skew_threshold}).")
        
        if high_kurt > 0:
            print(f"🏔️ Peaks: High Kurtosis detected in {high_kurt} columns (>{self.kurtosis_threshold}).")
        else:
            print("✨ Peaks: No extreme Kurtosis found. Healthy tails.")
            
        if missing_count > 0:
            print(f"⚠️ Null Values: {missing_count} columns contain missing data.")
        else:
            print("✨ Null Values: Dataset is complete (No missing data).")
        
        total_min, total_max = self.report['min'].min(), self.report['max'].max()
        print(f"📏 Data Range: {total_min:.2f} to {total_max:.2f}")
        
        print(f"🎯 Feature Verdict: {self.report[self.report['vortex_action'] == '✅ STRONG SIGNAL'].shape[0]} Strong Signals.")
        print("-" * 50)

    def get_report(self):
        print(f"\n🎯 --- TARGET ANALYSIS ({self.task.upper()}) ---")
        if self.task == 'classification':
            counts = self.y.value_counts()
            perms = (self.y.value_counts(normalize=True) * 100)
            print(pd.DataFrame({'Count': counts, 'Percentage': perms.map('{:.2f}%'.format)}))

        print("\n🧠 Scanning features for signal, noise, and AUC-ROC...")

        stats = pd.DataFrame(index=self.X.columns)
        stats['dtype'] = self.X.dtypes.astype(str)
        
        # Numeric base stats
        stats['mean'] = self.X.mean(numeric_only=True)
        stats['std'] = self.X.std(numeric_only=True)
        stats['min'] = self.X.min(numeric_only=True)
        stats['max'] = self.X.max(numeric_only=True)
        stats['skewness'] = self.X.skew(numeric_only=True).fillna(0.0)
        stats['kurtosis'] = self.X.kurtosis(numeric_only=True).fillna(0.0)
        stats['null_ratio'] = self.X.isnull().mean()
        stats['unique_counts'] = self.X.nunique()

        # FIXED LOOP: Correct column alignment
        for col in self.X.columns:
            if np.issubdtype(self.X[col].dtype, np.number):
                # 1. Outlier Count
                Q1, Q3 = self.X[col].quantile(0.25), self.X[col].quantile(0.75)
                IQR = Q3 - Q1
                stats.loc[col, 'outlier_count'] = ((self.X[col] < (Q1 - self.outlier_iqr_multiplier * IQR)) | 
                                                   (self.X[col] > (Q3 + self.outlier_iqr_multiplier * IQR))).sum()
                
                # 2. AUC-ROC (Area Under the Curve)
                if self.task == 'classification' and self.y.nunique() == 2:
                    try:
                        score = roc_auc_score(self.y, self.X[col])
                        stats.loc[col, 'auc_roc'] = max(score, 1 - score)
                    except:
                        stats.loc[col, 'auc_roc'] = 0.5
                else:
                    stats.loc[col, 'auc_roc'] = 0.0

                # 3. Absolute Correlation
                stats.loc[col, 'abs_target_corr'] = abs(self.X[col].corr(self.y))
            else:
                stats.loc[col, 'outlier_count'] = 0
                stats.loc[col, 'auc_roc'] = 0.5
                stats.loc[col, 'abs_target_corr'] = 0.0

        # LGBM Gain
        X_tmp = self.X.copy()
        for col in X_tmp.select_dtypes(exclude=[np.number]).columns:
            X_tmp[col] = X_tmp[col].astype('category')

        model = LGBMClassifier(n_estimators=100, importance_type='gain', verbosity=-1) if self.task == 'classification' else \
                LGBMRegressor(n_estimators=100, importance_type='gain', verbosity=-1)
            
        model.fit(X_tmp, self.y)
        stats['lgbm_gain'] = model.feature_importances_

        # Verdict
        def judge(row):
            if row['lgbm_gain'] > 100 or row['auc_roc'] > 0.65: return "✅ STRONG SIGNAL"
            if row['lgbm_gain'] == 0 and row['auc_roc'] <= 0.51: return "🗑️ GLOBAL NOISE"
            return "⚠️ WEAK SIGNAL"

        stats['vortex_action'] = stats.apply(judge, axis=1)
        
        # Explicit column order to prevent shifting
        final_cols = ['dtype', 'mean', 'std', 'min', 'max', 'skewness', 'kurtosis', 
                      'outlier_count', 'auc_roc', 'abs_target_corr', 'null_ratio', 
                      'unique_counts', 'lgbm_gain', 'vortex_action']
        
        self.report = stats[final_cols].sort_values(by='lgbm_gain', ascending=False)
        self._generate_text_summary()
        
        return self.report

    def plot_vortex_eda(self):
        if self.report is None: return
        cols = self.X.columns.tolist()
        n_rows = (len(cols) + 2) // 3
        plt.figure(figsize=(18, 5 * n_rows))
        for i, col in enumerate(cols):
            plt.subplot(n_rows, 3, i + 1)
            if self.X[col].nunique() > 10 and np.issubdtype(self.X[col].dtype, np.number):
                sns.histplot(self.X[col], kde=True, color='teal')
            else:
                sns.countplot(x=self.X[col], palette='viridis')
            plt.title(f"{col}\nGain: {self.report.loc[col, 'lgbm_gain']:.1f} | AUC: {self.report.loc[col, 'auc_roc']:.3f}")
        plt.tight_layout()
        plt.show()
