import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

class VortexIntelligence:
    def __init__(self, X, y, task='classification', imbalance_threshold=3.0, 
                 skew_threshold=1.0, kurtosis_threshold=10.0, outlier_iqr_multiplier=3.0):
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

        if outlier_cols > 0:
            print(f"🚩 Outliers: Detected in {outlier_cols} columns (Limit: {self.outlier_iqr_multiplier}xIQR)")
        else:
            print(f"✨ Outliers: It's fine. No extreme outliers found (Limit: {self.outlier_iqr_multiplier}xIQR)")

        if (right_skew + left_skew) > 0:
            print(f"📐 Skewness: {right_skew} Right, {left_skew} Left detected (Thr: ±{self.skew_threshold})")
        else:
            print(f"✨ Skewness: It's fine. All features are symmetric (Thr: ±{self.skew_threshold})")
        
        if high_kurt > 0:
            print(f"🏔️ Peaks: {high_kurt} columns with High Kurtosis (Thr: >{self.kurtosis_threshold})")
        else:
            print(f"✨ Peaks: It's fine. Healthy tails detected (Thr: <{self.kurtosis_threshold})")
            
        if missing_count > 0:
            print(f"⚠️ Null Values: Missing data in {missing_count} columns")
        else:
            print(f"✨ Null Values: It's fine. Dataset is complete (100% density)")
        
        print(f"📏 Data Range: {self.report['min'].min():.2f} to {self.report['max'].max():.2f}")
        print(f"🎯 Feature Verdict: {self.report[self.report['vortex_action'] == '✅ STRONG SIGNAL'].shape[0]} Strong Signals.")
        print("-" * 65)

    def get_report(self):
        # 1. Initialize DataFrame with Index
        stats = pd.DataFrame(index=self.X.columns)
        
        # 2. Strict Column-by-Column Assignment to prevent horizontal shifting
        for col in self.X.columns:
            is_num = np.issubdtype(self.X[col].dtype, np.number)
            stats.loc[col, 'dtype'] = str(self.X[col].dtype)
            stats.loc[col, 'mean'] = self.X[col].mean() if is_num else 0
            stats.loc[col, 'std'] = self.X[col].std() if is_num else 0
            stats.loc[col, 'min'] = self.X[col].min() if is_num else 0
            stats.loc[col, 'max'] = self.X[col].max() if is_num else 0
            stats.loc[col, 'skewness'] = self.X[col].skew() if is_num else 0
            stats.loc[col, 'kurtosis'] = self.X[col].kurtosis() if is_num else 0
            stats.loc[col, 'null_ratio'] = self.X[col].isnull().mean()
            stats.loc[col, 'unique_counts'] = self.X[col].nunique()

            if is_num:
                # Outliers
                Q1, Q3 = self.X[col].quantile(0.25), self.X[col].quantile(0.75)
                IQR = Q3 - Q1
                stats.loc[col, 'outlier_count'] = ((self.X[col] < (Q1 - self.outlier_iqr_multiplier * IQR)) | (self.X[col] > (Q3 + self.outlier_iqr_multiplier * IQR))).sum()
                
                # AUC-ROC (Crucial: Locked to 'auc_roc' column)
                if self.task == 'classification':
                    try:
                        score = roc_auc_score(self.y, self.X[col])
                        stats.loc[col, 'auc_roc'] = max(score, 1 - score)
                    except: stats.loc[col, 'auc_roc'] = 0.5
                
                # Correlation
                stats.loc[col, 'abs_target_corr'] = abs(self.X[col].corr(self.y))
            else:
                stats.loc[col, 'outlier_count'] = 0
                stats.loc[col, 'auc_roc'] = 0.5
                stats.loc[col, 'abs_target_corr'] = 0.0

        # 3. LightGBM Gain
        X_tmp = self.X.copy()
        for c in X_tmp.select_dtypes(exclude=[np.number]).columns: X_tmp[c] = X_tmp[c].astype('category')
        model = LGBMClassifier(n_estimators=100, verbosity=-1) if self.task == 'classification' else LGBMRegressor(n_estimators=100, verbosity=-1)
        model.fit(X_tmp, self.y)
        stats['lgbm_gain'] = model.feature_importances_

        # 4. Verdict Logic
        stats['vortex_action'] = stats.apply(lambda r: "✅ STRONG SIGNAL" if r['lgbm_gain'] > 100 or r['auc_roc'] > 0.65 else "⚠️ WEAK SIGNAL", axis=1)
        
        # 5. Force column order to fix the visual printout
        order = ['dtype', 'mean', 'std', 'min', 'max', 'skewness', 'kurtosis', 'outlier_count', 'auc_roc', 'abs_target_corr', 'null_ratio', 'unique_counts', 'lgbm_gain', 'vortex_action']
        self.report = stats[order].sort_values('lgbm_gain', ascending=False)
        
        self._generate_text_summary()
        return self.report
