import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class VortexIntelligence:
    # ANSI Color Constants
    GREEN, RED, YELLOW, CYAN = "\033[92m", "\033[91m", "\033[93m", "\033[96m"
    BOLD, RESET = "\033[1m", "\033[0m"

    def __init__(self, X, y, task='classification', 
                 imbalance_threshold=4, skew_threshold=0.75, 
                 kurtosis_threshold=10.0, outlier_iqr_multiplier=3.0,
                 redundancy_threshold=0.90, cardinality_threshold=100,
                 leakage_threshold=0.95, feature_names=None):
        
        if isinstance(X, np.ndarray):
            feature_names = feature_names or [f"feat_{i}" for i in range(X.shape[1])]
            self.X = pd.DataFrame(X, columns=feature_names).reset_index(drop=True)
        else:
            self.X = X.copy().reset_index(drop=True)
            
        self.y = pd.Series(y).reset_index(drop=True)
        self.task = task.lower()
        self.report = None
        
        # Thresholds
        self.imbalance_threshold = imbalance_threshold
        self.skew_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.redundancy_threshold = redundancy_threshold
        self.cardinality_threshold = cardinality_threshold
        self.leakage_threshold = leakage_threshold
        self.cat_features = []

    def _determine_data_level(self, col):
        unique_c = self.X[col].nunique()
        is_num = np.issubdtype(self.X[col].dtype, np.number)
        if not is_num or unique_c <= 5: return "Nominal"
        if is_num and np.array_equal(self.X[col], self.X[col].astype(int)) and unique_c <= 20: return "Ordinal"
        return "Ratio" if (is_num and self.X[col].min() >= 0) else "Interval"

    def _generate_text_summary(self):
        if self.report is None: return
        
        # Reports
        num_rep = self.report[self.report['level'].isin(['Interval', 'Ratio'])]
        cat_rep = self.report[self.report['level'].isin(['Nominal', 'Ordinal'])]
        
        # Quality Logic
        corr_matrix = self.X.select_dtypes(include=[np.number]).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        redundant_pairs = [column for column in upper.columns if any(upper[column] > self.redundancy_threshold)]
        low_variance = [col for col in self.X.columns if self.X[col].nunique() <= 1]
        high_card = [col for col in self.cat_features if (self.X[col].nunique() > self.cardinality_threshold)]
        
        # Power Metric Switch (AUC for Class / Spearman for Reg)
        p_metric = 'auc_roc' if self.task == 'classification' else 'spearman_corr'
        leakage_features = self.report[self.report[p_metric] > self.leakage_threshold]['feature_name'].tolist()

        # Stats
        outlier_cols = num_rep[num_rep['outlier_count'] > 0].shape[0]
        skew_count = num_rep[(num_rep['skewness'] != "Categorical") & (num_rep['skewness'].abs() > self.skew_threshold)].shape[0]
        high_kurt = num_rep[(num_rep['kurtosis'] != "Categorical") & (num_rep['kurtosis'] > self.kurtosis_threshold)].shape[0]
        strong_signals = self.report[self.report['vortex_action'] == '✅ STRONG SIGNAL'].shape[0]

        def b(text): return f"{self.BOLD}{text}{self.RESET}"
        pad = 25
        print(f"\n{b('--- VORTEX INTELLIGENCE SUMMARY ---')}\n")

        # 1-4 Foundation & Health
        print(f"📋 {b('1. Dataset Scale'):<{pad}} : {self.CYAN}{b(f'{len(self.X):,} Rows x {self.X.shape[1]} Features')}")
        print(f"📊 {b('2. Composition'):<{pad}} : {self.CYAN}{b(f'{num_rep.shape[0]} Numerical, {cat_rep.shape[0]} Categorical')}")
        print(f"✨ {b('3. Null Values'):<{pad}} : {(self.RED if self.report['null_ratio'].sum() > 0 else self.GREEN)}{b('Complete' if self.report['null_ratio'].sum() == 0 else 'Missing detected')}")
        if self.task == 'classification':
            ratio = self.y.value_counts().max() / self.y.value_counts().min()
            print(f"⚖️ {b('4. Balance'):<{pad}} : {(self.RED if ratio > self.imbalance_threshold else self.GREEN)}{b(f'Ratio {ratio:.2f}:1')}")

        # 5-7 Quality
        print(f"🖇️ {b('5. Redundancy'):<{pad}} : {(self.YELLOW if redundant_pairs else self.GREEN)}{b(f'{len(redundant_pairs)} duplicates' if redundant_pairs else 'Clean')} (>{self.redundancy_threshold})")
        print(f"🧊 {b('6. Stability'):<{pad}} : {(self.RED if low_variance else self.GREEN)}{b(f'{len(low_variance)} constant cols' if low_variance else 'Healthy variance')}")
        print(f"🗂️ {b('7. Cardinality'):<{pad}} : {(self.YELLOW if high_card else self.GREEN)}{b(f'High complexity in {len(high_card)} cols' if high_card else 'Well-grouped')} (>{self.cardinality_threshold})")

        # 8-12 Math & Ranges
        print(f"🚩 {b('8. Outliers'):<{pad}} : {(self.RED if outlier_cols > 0 else self.GREEN)}{b(outlier_cols if outlier_cols > 0 else 'None')} ({self.outlier_iqr_multiplier}xIQR)")
        print(f"📐 {b('9. Skewness'):<{pad}} : {(self.RED if skew_count > 0 else self.GREEN)}{b(skew_count)} (Thr: ±{self.skew_threshold})")
        print(f"🏔️ {b('10. Peaks (Kurtosis)'):<{pad}} : {(self.RED if high_kurt > 0 else self.GREEN)}{b(high_kurt)} (Thr: <{self.kurtosis_threshold})")
        print(f"📉 {b('11. Categorical Range'):<{pad}} : {(self.CYAN if cat_rep['min'].min() >= 0 else self.RED)}{b(f'Min: {cat_rep.min().min():.2f} | Max: {cat_rep.max().max():.2f}')}")
        print(f"📈 {b('12. Numerical Range'):<{pad}} : {self.CYAN}{b(f'Min: {num_rep.min().min():.2f} | Max: {num_rep.max().max():.2f}')}")

        # 13-15 Power & Verdict
        top_f, top_v = self.report.iloc[0]['feature_name'], self.report.iloc[0][p_metric]
        print(f"🛡️ {b('13. Predictive'):<{pad}} : {self.GREEN}{b(f'[{top_f}] {p_metric.upper()}: {top_v:.4f}')}")
        print(f"🌊 {b('14. Leakage Alert'):<{pad}} : {(self.RED if leakage_features else self.GREEN)}{b(f'SUSPICIOUS: {leakage_features}' if leakage_features else 'None')} (>{self.leakage_threshold})")
        
        v_color = self.RED if (leakage_features or low_variance or (not cat_rep.empty and cat_rep['min'].min() < 0)) else self.GREEN
        print(f"🎯 {b('15. Final Verdict'):<{pad}} : {v_color}{b(f'{strong_signals} Strong Signals. Check RED flags before jump.')}\n")

    def get_report(self):
        X_tmp = self.X.copy()
        self.cat_features = []
        corr_matrix = self.X.select_dtypes(include=[np.number]).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        for col in self.X.columns:
            if self._determine_data_level(col) in ["Nominal", "Ordinal"]:
                self.cat_features.append(col)
                X_tmp[col] = X_tmp[col].astype('category')
        
        model = LGBMClassifier(n_estimators=100, verbosity=-1) if self.task == 'classification' else LGBMRegressor(n_estimators=100, verbosity=-1)
        model.fit(X_tmp, self.y)
        gains = dict(zip(self.X.columns, model.feature_importances_))
        
        data_list = []
        for col in self.X.columns:
            level = self._determine_data_level(col)
            is_num = np.issubdtype(self.X[col].dtype, np.number)
            is_cat = level in ["Nominal", "Ordinal"]
            
            # Predictive Logic
            if self.task == 'classification':
                try: 
                    score = roc_auc_score(self.y, pd.to_numeric(self.X[col], errors='coerce').fillna(0))
                    power_val = max(score, 1 - score)
                except: power_val = 0.5
                metric_name = 'auc_roc'
            else:
                power_val = self.X[col].corr(self.y, method='spearman')
                power_val = abs(power_val) if not np.isnan(power_val) else 0.0
                metric_name = 'spearman_corr'

            # Stats logic
            if is_num:
                stats = self.X[col].describe()
                skew_v, kurt_v = (self.X[col].skew(), self.X[col].kurtosis()) if not is_cat else ("Categorical", "Categorical")
                out_c = ((self.X[col] < (stats['25%'] - self.outlier_iqr_multiplier*(stats['75%']-stats['25%']))) | 
                         (self.X[col] > (stats['75%'] + self.outlier_iqr_multiplier*(stats['75%']-stats['25%'])))).sum() if not is_cat else -1
            else:
                stats = {'mean':0,'std':0,'min':0,'25%':0,'50%':0,'75%':0,'max':0}
                skew_v = kurt_v = "Categorical"; out_c = -1

            twins = upper.index[upper[col] > self.redundancy_threshold].tolist()

            data_list.append({
                'feature_name': col, 'level': level, 'importance_gain': gains.get(col, 0),
                metric_name: power_val, 'vortex_action': "✅ STRONG SIGNAL" if (gains.get(col, 0) > 100 or power_val > 0.65) else "⚠️ WEAK SIGNAL",
                'duplicate_of': ", ".join(twins) if twins else "None", 'is_leakage': power_val > self.leakage_threshold,
                'is_constant': self.X[col].nunique() <= 1, 'cardinality': self.X[col].nunique(), 'null_ratio': self.X[col].isnull().mean(),
                'mean': stats['mean'], 'std_dev': stats['std'], 'min': stats['min'], 'p25': stats['25%'], 'p50': stats['50%'], 'p75': stats['75%'], 'max': stats['max'],
                'skewness': skew_v, 'kurtosis': kurt_v, 'outlier_count': int(out_c)
            })
            
        self.report = pd.DataFrame(data_list).sort_values(metric_name, ascending=False).reset_index(drop=True)
        self._generate_text_summary()
        return self.report
    def get_visual_report(self, figsize=(18, 5)):
        if self.report is None: 
            self.get_report()
        else:
            self._generate_text_summary()
        sns.set_style("whitegrid")
        
        for feature in self.report['feature_name']:
            u_count = self.X[feature].nunique()
            is_num = np.issubdtype(self.X[feature].dtype, np.number)
            is_cont = is_num and u_count > 10
            
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            fig.suptitle(f"FEATURE ANALYSIS: {feature.upper()}", fontsize=16, fontweight='bold', y=1.05)
            
            # --- 1. DISTRIBUTION ---
            if is_cont:
                sns.histplot(self.X[feature], kde=True, ax=axes[0], color='skyblue')
                axes[0].set_title(f"Distribution (Skew: {self.X[feature].skew():.2f})")
            else:
                sns.countplot(data=self.X, x=feature, ax=axes[0], palette="viridis")
                axes[0].set_title(f"Frequency (Unique: {u_count})")
            
            # --- 2. OUTLIERS / COMPOSITION ---
            if is_cont:
                sns.boxplot(x=self.X[feature], ax=axes[1], color='salmon', fliersize=5)
                axes[1].set_title("Outlier Spread (Boxplot)")
            else:
                # Handle High Cardinality in Pie Chart (Show Top 10 + Other)
                v_counts = self.X[feature].value_counts()
                if len(v_counts) > 10:
                    display_counts = v_counts.head(10)
                    display_counts['Other'] = v_counts.iloc[10:].sum()
                else:
                    display_counts = v_counts
                
                display_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                    colors=sns.color_palette("pastel"), startangle=90)
                axes[1].set_ylabel('')
                axes[1].set_title(f"Composition (Top {min(10, len(v_counts))})")
            
            # --- 3. PREDICTIVE POWER (The 'Leakage' Visualizer) ---
            if self.task == 'classification':
                if not is_cont:
                    # Heatmap of correlation with target
                    ct = pd.crosstab(self.X[feature], self.y, normalize='index')
                    sns.heatmap(ct, annot=True, cmap="YlGnBu", ax=axes[2], cbar=False, fmt='.2f')
                    axes[2].set_title("Target Purity (Heatmap)")
                else:
                    # Violin plot shows if distributions are totally separated (Leakage Indicator)
                    sns.violinplot(x=self.y, y=self.X[feature], ax=axes[2], palette="muted", split=True)
                    axes[2].set_title("Target Separation (Violin)")
            
            plt.tight_layout()
            plt.show()
            return self.report
