import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import skew, kurtosis
import warnings

warnings.filterwarnings('ignore')

class VortexIntelligence:
    def __init__(self, X, y, task='classification'):
        self.X = X
        self.y = y
        self.task = task.lower()
        self.report = None
        
        # Display settings for professional output
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', 1000)

    def _generate_text_summary(self):
        """Generates the descriptive text intelligence summary."""
        if self.report is None:
            print("💡 To view the dataframe report, use .report. Not calculated yet, use get_report() to calculate.")
            return

        # Feature Counts
        num_cols = self.X.select_dtypes(include=[np.number]).shape[1]
        cat_cols = self.X.select_dtypes(exclude=[np.number]).shape[1]

        # Outlier & Skew Logic (Numerical only)
        num_stats = self.report[self.report['dtype'].str.contains('int|float')]
        outlier_cols = num_stats[num_stats['outlier_count'] > 0].shape[0]
        right_skew = num_stats[num_stats['skewness'] > 0.5].shape[0]
        left_skew = num_stats[num_stats['skewness'] < -0.5].shape[0]
        high_kurt = num_stats[num_stats['kurtosis'] > 3].shape[0]
        low_kurt = num_stats[num_stats['kurtosis'] < -1].shape[0]
        
        null_cols = self.report[self.report['null_ratio'] > 0].shape[0]
        global_min = self.X.select_dtypes(include=[np.number]).min().min()
        global_max = self.X.select_dtypes(include=[np.number]).max().max()

        print("\n📝 --- VORTEX INTELLIGENCE SUMMARY ---")
        
        # Target Summary Integration
        y_series = pd.Series(self.y)
        if self.task == 'classification':
            counts = y_series.value_counts()
            ratio = counts.max() / counts.min()
            status = "High Imbalance" if ratio > 3 else "Balanced"
            print(f"⚖️ Balance: {status} detected (Ratio {ratio:.2f}:1).")
        else:
            print(f"📈 Target Profile: Skew {y_series.skew():.2f} | Kurtosis {y_series.kurtosis():.2f}")

        print(f"📂 Structure: {num_cols} Numerical columns and {cat_cols} Categorical columns found.")
        
        if outlier_cols > 0:
            print(f"🚩 Outliers: Outliers found in {outlier_cols} columns.")
        else:
            print("✨ Outliers: No outliers found in numerical columns.")
            
        print(f"📐 Skewness: {right_skew} columns show Right Skewness, {left_skew} columns show Left Skewness.")
        print(f"🏔️ Peaks: {high_kurt} columns show High Kurtosis (Heavy Tails), {low_kurt} columns show Low Kurtosis.")
        
        if null_cols > 0:
            print(f"☁️ Null Values: Detected in {null_cols} columns.")
        else:
            print("✨ Null Values: No null values found in the dataset.")
            
        print(f"📏 Data Range: Numerical data ranges from {global_min} to {global_max}.")
        print(f"🎯 Feature Verdict: {self.report[self.report['vortex_action'] == '✅ STRONG SIGNAL'].shape[0]} features identified as Strong Signals.")
        print("-" * 40)
        print("💡 To view the full dataframe report, use .report")

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
