import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

class VortexIntelligence:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def __init__(self, X, y, task='classification', 
                 imbalance_threshold=4, 
                 skew_threshold=0.75, 
                 kurtosis_threshold=10.0, 
                 outlier_iqr_multiplier=3.0,
                 redundancy_threshold=0.90,    # New Parameter
                 cardinality_threshold=100,    # New Parameter
                 leakage_threshold=0.95,       # New Parameter
                 feature_names=None):
        
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"feat_{i}" for i in range(X.shape[1])]
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
        
        numeric_report = self.report[self.report['level'].isin(['Interval', 'Ratio'])]
        categorical_report = self.report[self.report['level'].isin(['Nominal', 'Ordinal'])]
        
        # 1. Redundancy Logic
        corr_matrix = self.X.select_dtypes(include=[np.number]).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        redundant_pairs = [column for column in upper.columns if any(upper[column] > self.redundancy_threshold)]
        
        # 2. Stability Logic
        low_variance = [col for col in self.X.columns if self.X[col].nunique() <= 1]
        
        # 3. Cardinality Logic
        high_card = [col for col in self.cat_features if (self.X[col].nunique() > self.cardinality_threshold)]
        
        # 4. Leakage Logic
        leakage_features = self.report[self.report['auc_roc'] > self.leakage_threshold]['feature_name'].tolist()

        # Stat Aggregations
        outlier_cols = numeric_report[numeric_report['outlier_count'] > 0].shape[0]
        skew_count = numeric_report[(numeric_report['skewness'] != "Categorical") & 
                                    ((numeric_report['skewness'] > self.skew_threshold) |
                                     (numeric_report['skewness'] < -self.skew_threshold))].shape[0]
        high_kurt = numeric_report[(numeric_report['kurtosis'] != "Categorical") & 
                                   (numeric_report['kurtosis'] > self.kurtosis_threshold)].shape[0]
        missing_count = self.report[self.report['null_ratio'] > 0].shape[0]
        levels = self.report['level'].value_counts().to_dict()
        strong_signals = self.report[self.report['vortex_action'] == '✅ STRONG SIGNAL'].shape[0]

        num_min, num_max = (numeric_report['min'].min(), numeric_report['max'].max()) if not numeric_report.empty else (0,0)
        cat_min, cat_max = (categorical_report['min'].min(), categorical_report['max'].max()) if not categorical_report.empty else (0,0)

        def b(text): return f"{self.BOLD}{text}{self.RESET}"
        pad = 25

        print(f"\n{b('--- VORTEX INTELLIGENCE SUMMARY ---')}\n")

        # --- FOUNDATION ---
        print(f"📋 {b('1. Dataset Scale'):<{pad}} : {self.CYAN}{b(f'{len(self.X):,} Rows x {self.X.shape[1]} Features')}")
        num_count = levels.get('Ratio', 0) + levels.get('Interval', 0)
        cat_count = levels.get('Nominal', 0) + levels.get('Ordinal', 0)
        print(f"📊 {b('2. Composition'):<{pad}} : {self.CYAN}{b(f'{num_count} Numerical, {cat_count} Categorical')}")

        # --- HEALTH ---
        print(f"✨ {b('3. Null Values'):<{pad}} : {(self.RED if missing_count > 0 else self.GREEN)}{b('Missing detected' if missing_count > 0 else 'Dataset complete (100% density)')}")
        if self.task == 'classification':
            counts = self.y.value_counts()
            ratio = counts.max() / counts.min()
            bal_color = self.RED if ratio > self.imbalance_threshold else self.GREEN
            print(f"⚖️ {b('4. Balance'):<{pad}} : {bal_color}{b(f'Ratio {ratio:.2f}:1 (Thr: {self.imbalance_threshold}:1)')}")

        # --- QUALITY ---
        redun_color = self.YELLOW if redundant_pairs else self.GREEN
        print(f"🖇️ {b('5. Redundancy'):<{pad}} : {redun_color}{b(f'{len(redundant_pairs)} duplicates' if redundant_pairs else 'No redundancy')} {b(f'(Thr: >{self.redundancy_threshold})')}")
        
        stab_color = self.RED if low_variance else self.GREEN
        print(f"🧊 {b('6. Stability'):<{pad}} : {stab_color}{b(f'{len(low_variance)} constant cols' if low_variance else 'Healthy variance')}")
        
        card_color = self.YELLOW if high_card else self.GREEN
        print(f"🗂️ {b('7. Cardinality'):<{pad}} : {card_color}{b(f'High complexity in {len(high_card)} cols' if high_card else 'Well-grouped')} {b(f'(Limit: {self.cardinality_threshold})')}")

        # --- MATH ---
        print(f"🚩 {b('8. Outliers'):<{pad}} : {(self.RED if outlier_cols > 0 else self.GREEN)}{b(f'Found in {outlier_cols} cols' if outlier_cols > 0 else 'No extreme outliers')} {b(f'(Limit: {self.outlier_iqr_multiplier}xIQR)')}")
        print(f"📐 {b('9. Skewness'):<{pad}} : {(self.RED if skew_count > 0 else self.GREEN)}{b(f'{skew_count} cols skewed' if skew_count > 0 else 'Symmetric')} {b(f'(Thr: ±{self.skew_threshold})')}")
        print(f"🏔️ {b('10. Peaks (Kurtosis)'):<{pad}} : {(self.RED if high_kurt > 0 else self.GREEN)}{b(f'{high_kurt} high-peak cols' if high_kurt > 0 else 'Healthy tails')} {b(f'(Thr: <{self.kurtosis_threshold})')}")

        # --- RANGES ---
        cat_range_color = self.CYAN if cat_min >= 0 else self.RED
        print(f"📉 {b('11. Categorical Range'):<{pad}} : {cat_range_color}{b(f'Min: {cat_min:.2f} | Max: {cat_max:.2f}')}")
        print(f"📈 {b('12. Numerical Range'):<{pad}} : {self.CYAN}{b(f'Min: {num_min:.2f} | Max: {num_max:.2f}')}")

        # --- POWER ---
        top_f = self.report.iloc[0]['feature_name']
        auc_val = self.report.iloc[0]['auc_roc']
        print(f"🛡️ {b('13. Predictive (AUC)'):<{pad}} : {self.GREEN if auc_val > 0.65 else self.YELLOW}{b(f'Top: [{top_f}] AUC: {auc_val:.4f}')}")
        
        leak_color = self.RED if leakage_features else self.GREEN
        print(f"🌊 {b('14. Leakage Alert'):<{pad}} : {leak_color}{b(f'SUSPICIOUS: {leakage_features}' if leakage_features else 'No leakage detected')} {b(f'(Thr: >{self.leakage_threshold})')}")

        # --- VERDICT ---
        verdict_color = self.RED if (leakage_features or low_variance or cat_min < 0) else (self.YELLOW if (redundant_pairs or high_card) else self.GREEN)
        print(f"🎯 {b('15. Final Verdict'):<{pad}} : {verdict_color}{b(f'{strong_signals} Strong Signals. Review RED flags before training.')}\n")

    def get_report(self):
        X_tmp = self.X.copy()
        self.cat_features = []
        for col in self.X.columns:
            level = self._determine_data_level(col)
            if level in ["Nominal", "Ordinal"]:
                self.cat_features.append(col)
                X_tmp[col] = X_tmp[col].astype('category')
        
        model = LGBMClassifier(n_estimators=100, importance_type='gain', verbosity=-1) if self.task == 'classification' else LGBMRegressor(n_estimators=100, importance_type='gain', verbosity=-1)
        model.fit(X_tmp, self.y)
        gains = dict(zip(self.X.columns, model.feature_importances_))
        
        data_list = []
        for col in self.X.columns:
            level = self._determine_data_level(col)
            is_num = np.issubdtype(self.X[col].dtype, np.number)
            skew_val = self.X[col].skew() if is_num else "Categorical"
            kurt_val = self.X[col].kurtosis() if is_num else "Categorical"
            
            out_count = -1
            if is_num and level in ["Interval", "Ratio"]:
                Q1, Q3 = self.X[col].quantile(0.25), self.X[col].quantile(0.75)
                out_count = ((self.X[col] < (Q1 - self.outlier_iqr_multiplier * (Q3-Q1))) | (self.X[col] > (Q3 + self.outlier_iqr_multiplier * (Q3-Q1)))).sum()
            
            auc = 0.5
            if self.task == 'classification':
                try:
                    score = roc_auc_score(self.y, pd.to_numeric(self.X[col], errors='coerce').fillna(0))
                    auc = max(score, 1 - score)
                except: auc = 0.5
            
            data_list.append({
                'feature_name': col, 'level': level, 
                'min': self.X[col].min() if is_num else 0, 'max': self.X[col].max() if is_num else 0, 
                'skewness': skew_val, 'kurtosis': kurt_val, 
                'outlier_count': int(out_count), 'auc_roc': auc, 
                'null_ratio': self.X[col].isnull().mean(), 
                'vortex_action': "✅ STRONG SIGNAL" if (gains.get(col, 0) > 100 or auc > 0.65) else "⚠️ WEAK SIGNAL"
            })
            
        self.report = pd.DataFrame(data_list).sort_values('auc_roc', ascending=False).reset_index(drop=True)
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
