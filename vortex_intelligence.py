import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import skew, kurtosis
import warnings

warnings.filterwarnings('ignore')

class VortexIntelligence:
    def __init__(self, X, y, task='classification', 
                 imbalance_threshold=3.0, 
                 skew_threshold=0.5, 
                 kurtosis_threshold=3.0, 
                 outlier_iqr_multiplier=1.5):
        
        self.X = X.copy().reset_index(drop=True)
        self.y = pd.Series(y).reset_index(drop=True)
        self.task = task.lower()
        self.report = None
        
        # --- Configurable Thresholds (No more hard-coding) ---
        self.imbalance_threshold = imbalance_threshold
        self.skew_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.outlier_iqr_multiplier = outlier_iqr_multiplier

    def _generate_text_summary(self):
        if self.report is None: return

        # 1. Gather stats using dynamic thresholds
        num_stats = self.report[self.report['dtype'].str.contains('int|float|complex')]
        outlier_cols = num_stats[num_stats['outlier_count'] > 0].shape[0]
        right_skew = num_stats[num_stats['skewness'] > self.skew_threshold].shape[0]
        left_skew = num_stats[num_stats['skewness'] < -self.skew_threshold].shape[0]
        high_kurt = num_stats[num_stats['kurtosis'] > self.kurtosis_threshold].shape[0]
        low_kurt = num_stats[num_stats['kurtosis'] < -1.0].shape[0]
        null_cols = self.report[self.report['null_ratio'] > 0].shape[0]
        
        num_data = self.X.select_dtypes(include=[np.number])

        print("\n📝 --- VORTEX INTELLIGENCE SUMMARY ---")
        
        # --- Target Balance ---
        if self.task == 'classification':
            counts = self.y.value_counts()
            ratio = counts.max() / counts.min()
            if ratio > self.imbalance_threshold:
                print(f"⚖️ Balance: High Imbalance detected (Ratio {ratio:.2f}:1).")
            else:
                print(f"⚖️ Balance: Classes are well-balanced (Ratio {ratio:.2f}:1).")
        
        # --- Outliers ---
        if outlier_cols > 0:
            print(f"🚩 Outliers: Detected in {outlier_cols} columns (Threshold: {self.outlier_iqr_multiplier}xIQR).")
        else:
            print("✨ Outliers: No outliers detected. Distribution is stable.")

        # --- Skewness ---
        if (right_skew + left_skew) > 0:
            print(f"📐 Skewness: {right_skew} Right, {left_skew} Left detected (Threshold: ±{self.skew_threshold}).")
        else:
            print("✨ Skewness: All features are mathematically symmetrical.")

        # --- Peaks (Kurtosis) ---
        if high_kurt > 0:
            print(f"🏔️ Peaks: High Kurtosis detected in {high_kurt} columns (Threshold: >{self.kurtosis_threshold}).")
        else:
            print("✨ Peaks: No extreme Kurtosis found. Tails are healthy.")

        # --- Null Values ---
        if null_cols > 0:
            print(f"☁️ Null Values: Detected in {null_cols} columns.")
        else:
            print("✨ Null Values: Dataset is complete (No missing data).")
            
        print(f"📏 Data Range: {num_data.min().min():.2f} to {num_data.max().max():.2f}")
        print(f"🎯 Feature Verdict: {self.report[self.report['vortex_action'] == '✅ STRONG SIGNAL'].shape[0]} Strong Signals.")
        print("-" * 40)

    def _analyze_target(self):
        print(f"\n🎯 --- TARGET ANALYSIS ({self.task.upper()}) ---")
        y_series = pd.Series(self.y)
        
        if self.task == 'classification':
            counts = y_series.value_counts()
            perms = y_series.value_counts(normalize=True) * 100
            balance_df = pd.DataFrame({'Count': counts, 'Percentage': perms.map('{:.2f}%'.format)})
            print("Class Balance:")
            print(balance_df)
            print(f"\nBalance Ratio: {counts.max() / counts.min():.2f}:1")
        else:
            print(f"Mean: {y_series.mean():.4f} | Median: {y_series.median():.4f}")
            print(f"Skew: {y_series.skew():.4f} | Kurtosis: {y_series.kurtosis():.4f}")

    def get_report(self):
        self._analyze_target()
        print("\n🧠 Scanning features for signal and noise...")

        # Initialize stats with data types
        stats = pd.DataFrame(index=self.X.columns)
        stats['dtype'] = self.X.dtypes.astype(str)
        
        # Descriptive base (including categorical via describe(include='all'))
        desc = self.X.describe(include='all').T
        stats = stats.join(desc[['mean', 'std', 'min', 'max']])

        # Advanced math (Numerical only)
        num_cols = self.X.select_dtypes(include=[np.number]).columns
        stats.loc[num_cols, 'skewness'] = self.X[num_cols].apply(lambda x: skew(x.dropna()))
        stats.loc[num_cols, 'kurtosis'] = self.X[num_cols].apply(lambda x: kurtosis(x.dropna()))
        
        # IQR Outliers
        Q1, Q3 = self.X[num_cols].quantile(0.25), self.X[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        stats.loc[num_cols, 'outlier_count'] = ((self.X[num_cols] < (Q1 - 1.5 * IQR)) | (self.X[num_cols] > (Q3 + 1.5 * IQR))).sum()

        # General stats
        stats['null_ratio'] = self.X.isnull().mean()
        stats['unique_counts'] = self.X.nunique()

        # Importance (LGBM handles categorical if we cast them to category type)
        X_tmp = self.X.copy()
        for col in X_tmp.select_dtypes(exclude=[np.number]).columns:
            X_tmp[col] = X_tmp[col].astype('category')

        model = LGBMClassifier(n_estimators=500, importance_type='gain', verbosity=-1) if self.task == 'classification' else LGBMRegressor(n_estimators=500, importance_type='gain', verbosity=-1)
        model.fit(X_tmp, self.y)
        stats['lgbm_gain'] = model.feature_importances_
        
        # Correlation (Numerical only)
        stats.loc[num_cols, 'abs_target_corr'] = self.X[num_cols].corrwith(pd.Series(self.y)).abs()

        # Verdict
        def judge(row):
            if row['lgbm_gain'] > 100 or (pd.notnull(row['abs_target_corr']) and row['abs_target_corr'] > 0.05):
                return "✅ STRONG SIGNAL"
            if row['lgbm_gain'] == 0 and (pd.isnull(row['abs_target_corr']) or row['abs_target_corr'] < 0.005):
                return "🗑️ GLOBAL NOISE"
            return "⚠️ WEAK SIGNAL"

        stats['vortex_action'] = stats.apply(judge, axis=1)
        self.report = stats.sort_values(by='lgbm_gain', ascending=False)
        print("✅ Intelligence Report Generated.")
        self._generate_text_summary()
        return self.report

    def plot_vortex_eda(self):
        """Visualizes all features in a 3-column grid."""
        if self.report is None: return print("Please run get_report() first.")
        cols = self.X.columns.tolist()
        total_plots = len(cols) + 1
        n_rows = (total_plots + 2) // 3
        plt.figure(figsize=(18, 5 * n_rows))
        
        for i, col in enumerate(cols):
            plt.subplot(n_rows, 3, i + 1)
            if self.X[col].nunique() > 10 and self.X[col].dtype != 'object':
                sns.histplot(self.X[col], kde=True, color='teal')
            else:
                sns.countplot(x=self.X[col], palette='viridis')
            plt.title(f"{col} ({self.X[col].dtype})\nGain: {self.report.loc[col, 'lgbm_gain']:.1f}")

        plt.subplot(n_rows, 3, total_plots)
        if self.task == 'classification':
            sns.countplot(x=self.y, palette='magma')
        else:
            sns.histplot(self.y, kde=True, color='orange')
        plt.title("Target (y) Distribution")
        plt.tight_layout()
        plt.show()
