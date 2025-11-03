import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# 设置样式，使用英文避免字体问题
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

class TrainDataEDA:
    def __init__(self, file_path):
        """Initialize EDA class"""
        self.file_path = file_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load data"""
        print("=" * 60)
        print("🚗 Used Car Training Data - Exploratory Data Analysis (EDA)")
        print("=" * 60)
        
        try:
            self.df = pd.read_csv(self.file_path, sep=' ')
            print(f"✅ Data loaded successfully!")
            print(f"📈 Data shape: {self.df.shape}")
            print(f"💰 Target variable: price (Used car price)")
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return
    
    def basic_info(self):
        """Basic information analysis"""
        print("\n" + "="*50)
        print("1️⃣  Basic Information Analysis")
        print("="*50)
        
        print(f"Dataset size: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n📋 Data type distribution:")
        dtype_counts = self.df.dtypes.value_counts()
        print(dtype_counts)
        
        print("\n📊 First 5 rows:")
        print(self.df.head())
        
        print("\n📊 Basic statistics:")
        print(self.df.describe())
    
    def target_variable_analysis(self):
        """Target variable (price) analysis"""
        print("\n" + "="*50)
        print("2️⃣  Target Variable (Price) Analysis - Key Focus")
        print("="*50)
        
        price = self.df['price']
        
        print("💰 Price basic statistics:")
        print(f"Average price: {price.mean():.2f} Yuan")
        print(f"Median price: {price.median():.2f} Yuan")
        print(f"Price std: {price.std():.2f} Yuan")
        print(f"Min price: {price.min():.2f} Yuan")
        print(f"Max price: {price.max():.2f} Yuan")
        print(f"Price range: {price.max() - price.min():.2f} Yuan")
        
        # Price quantiles
        print(f"\n📊 Price quantiles:")
        for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            print(f"{q*100:4.0f}% quantile: {price.quantile(q):8.2f} Yuan")
        
        # Price distribution visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Price histogram
        axes[0,0].hist(price, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Price Distribution')
        axes[0,0].set_xlabel('Price (Yuan)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(price.mean(), color='red', linestyle='--', label=f'Mean: {price.mean():.0f}')
        axes[0,0].axvline(price.median(), color='green', linestyle='--', label=f'Median: {price.median():.0f}')
        axes[0,0].legend()
        
        # Price box plot
        axes[0,1].boxplot(price)
        axes[0,1].set_title('Price Box Plot')
        axes[0,1].set_ylabel('Price (Yuan)')
        
        # Log price distribution
        log_price = np.log1p(price)
        axes[1,0].hist(log_price, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1,0].set_title('Log Price Distribution')
        axes[1,0].set_xlabel('log(Price+1)')
        axes[1,0].set_ylabel('Frequency')
        
        # Price cumulative distribution
        sorted_price = np.sort(price)
        y = np.arange(1, len(sorted_price) + 1) / len(sorted_price)
        axes[1,1].plot(sorted_price, y, color='purple')
        axes[1,1].set_title('Price Cumulative Distribution')
        axes[1,1].set_xlabel('Price (Yuan)')
        axes[1,1].set_ylabel('Cumulative Probability')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('price_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Price skewness and kurtosis
        skewness = stats.skew(price)
        kurtosis = stats.kurtosis(price)
        print(f"\n📈 Price distribution characteristics:")
        print(f"Skewness: {skewness:.3f} {'(Right-skewed)' if skewness > 0 else '(Left-skewed)' if skewness < 0 else '(Symmetric)'}")
        print(f"Kurtosis: {kurtosis:.3f} {'(Leptokurtic)' if kurtosis > 0 else '(Platykurtic)' if kurtosis < 0 else '(Mesokurtic)'}")
        
        # Price range analysis
        print(f"\n💰 Price range analysis:")
        price_bins = [0, 5000, 10000, 20000, 50000, float('inf')]
        price_labels = ['<5K', '5K-10K', '10K-20K', '20K-50K', '>50K']
        price_groups = pd.cut(price, bins=price_bins, labels=price_labels, right=False)
        price_dist = price_groups.value_counts().sort_index()
        
        for label, count in price_dist.items():
            percentage = count / len(price) * 100
            print(f"{label:>8}: {count:6d} cars ({percentage:5.1f}%)")
    
    def missing_values_analysis(self):
        """Missing values analysis"""
        print("\n" + "="*50)
        print("3️⃣  Missing Values Analysis")
        print("="*50)
        
        # Calculate missing values
        missing_data = pd.DataFrame({
            'Missing Count': self.df.isnull().sum(),
            'Missing Percentage(%)': (self.df.isnull().sum() / len(self.df)) * 100
        })
        missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_data) > 0:
            print("🔍 Fields with missing values:")
            print(missing_data)
            
            # Visualize missing values
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            missing_data['Missing Count'].plot(kind='bar', color='coral')
            plt.title('Missing Values Count by Field')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            missing_data['Missing Percentage(%)'].plot(kind='bar', color='orange')
            plt.title('Missing Values Percentage by Field (%)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("✅ No standard missing values (NaN) in dataset")
        
        # Check special values (like '-')
        print("\n🔍 Check special values:")
        special_values = {}
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or col == 'notRepairedDamage':
                unique_vals = self.df[col].unique()
                if '-' in unique_vals or len(unique_vals) < 20:
                    special_values[col] = unique_vals
                    print(f"{col}: {unique_vals}")
        
        # Analyze notRepairedDamage field specifically
        if 'notRepairedDamage' in self.df.columns:
            print(f"\n🔧 notRepairedDamage field analysis:")
            damage_counts = self.df['notRepairedDamage'].value_counts()
            print(damage_counts)
            
            # Analyze price by damage status
            if '-' in self.df['notRepairedDamage'].values:
                print(f"\n💰 Price analysis by damage status:")
                for damage_type in self.df['notRepairedDamage'].unique():
                    subset = self.df[self.df['notRepairedDamage'] == damage_type]['price']
                    print(f"{damage_type}: Avg {subset.mean():.0f} Yuan, Median {subset.median():.0f} Yuan, Count {len(subset)}")
    
    def feature_price_relationship(self):
        """Feature-price relationship analysis"""
        print("\n" + "="*50)
        print("4️⃣  Feature-Price Relationship Analysis")
        print("="*50)
        
        # Numerical features correlation with price
        numerical_features = ['power', 'kilometer', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox']
        existing_features = [f for f in numerical_features if f in self.df.columns]
        
        print("📊 Correlation coefficients between numerical features and price:")
        correlations = []
        for feature in existing_features:
            corr = self.df[feature].corr(self.df['price'])
            correlations.append((feature, corr))
            print(f"{feature:>12}: {corr:7.3f}")
        
        # Visualize correlations
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        features, corr_values = zip(*correlations)
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if x < 0 else 'blue' for x in corr_values]
        plt.barh(features, corr_values, color=colors, alpha=0.7)
        plt.title('Feature-Price Correlation Coefficients')
        plt.xlabel('Correlation Coefficient')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_price_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Scatter plots for important features
        important_features = ['power', 'kilometer']
        existing_important = [f for f in important_features if f in self.df.columns]
        
        if existing_important:
            fig, axes = plt.subplots(1, len(existing_important), figsize=(6*len(existing_important), 5))
            if len(existing_important) == 1:
                axes = [axes]
            
            for i, feature in enumerate(existing_important):
                axes[i].scatter(self.df[feature], self.df['price'], alpha=0.5, s=1)
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Price')
                axes[i].set_title(f'{feature} vs Price')
                
                # Add trend line
                z = np.polyfit(self.df[feature].dropna(), 
                              self.df.loc[self.df[feature].dropna().index, 'price'], 1)
                p = np.poly1d(z)
                axes[i].plot(self.df[feature], p(self.df[feature]), "r--", alpha=0.8)
            
            plt.tight_layout()
            plt.savefig('feature_price_scatter.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def categorical_analysis(self):
        """Categorical variables analysis"""
        print("\n" + "="*50)
        print("5️⃣  Categorical Variables Analysis")
        print("="*50)
        
        categorical_vars = ['brand', 'bodyType', 'fuelType', 'gearbox', 'seller', 'offerType']
        existing_cats = [var for var in categorical_vars if var in self.df.columns]
        
        for var in existing_cats:
            print(f"\n📊 {var} analysis:")
            
            # Basic distribution
            value_counts = self.df[var].value_counts()
            print(f"Number of categories: {len(value_counts)}")
            print("Top 10 categories distribution:")
            print(value_counts.head(10))
            
            # Average price by category
            avg_price_by_category = self.df.groupby(var)['price'].agg(['mean', 'median', 'count']).round(2)
            avg_price_by_category = avg_price_by_category.sort_values('mean', ascending=False)
            
            print(f"\n💰 Price statistics by {var} category (sorted by average price):")
            print(avg_price_by_category.head(10))
            
            # Visualization - show top 15 categories only
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Category distribution
            top_categories = value_counts.head(15)
            top_categories.plot(kind='bar', ax=axes[0])
            axes[0].set_title(f'{var} Distribution (Top 15)')
            axes[0].set_xlabel(var)
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Average price by category
            top_price_categories = avg_price_by_category.head(15)
            top_price_categories['mean'].plot(kind='bar', ax=axes[1], color='orange')
            axes[1].set_title(f'Average Price by {var} (Top 15)')
            axes[1].set_xlabel(var)
            axes[1].set_ylabel('Average Price (Yuan)')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{var}_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def time_analysis(self):
        """Time-related analysis"""
        print("\n" + "="*50)
        print("6️⃣  Time-Related Analysis")
        print("="*50)
        
        # Process registration date
        if 'regDate' in self.df.columns:
            # Convert date format
            self.df['regDate_str'] = self.df['regDate'].astype(str)
            self.df['reg_year'] = self.df['regDate_str'].str[:4].astype(int)
            self.df['reg_month'] = self.df['regDate_str'].str[4:6].astype(int)
            
            print("📅 Registration year distribution:")
            year_dist = self.df['reg_year'].value_counts().sort_index()
            print(year_dist)
            
            # Calculate car age
            current_year = 2020  # Assume current year is 2020
            self.df['car_age'] = current_year - self.df['reg_year']
            
            print(f"\n🚗 Car age statistics:")
            print(self.df['car_age'].describe())
            
            # Car age vs price relationship
            print(f"\n💰 Car age vs price relationship:")
            age_price_corr = self.df['car_age'].corr(self.df['price'])
            print(f"Car age vs price correlation: {age_price_corr:.3f}")
            
            # Visualize time analysis
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Registration year distribution
            year_dist.plot(kind='bar', ax=axes[0,0])
            axes[0,0].set_title('Vehicle Registration Year Distribution')
            axes[0,0].set_xlabel('Registration Year')
            axes[0,0].set_ylabel('Vehicle Count')
            
            # Car age distribution
            self.df['car_age'].hist(bins=30, ax=axes[0,1], alpha=0.7)
            axes[0,1].set_title('Car Age Distribution')
            axes[0,1].set_xlabel('Car Age (Years)')
            axes[0,1].set_ylabel('Frequency')
            
            # Car age vs price scatter plot
            axes[1,0].scatter(self.df['car_age'], self.df['price'], alpha=0.5, s=1)
            axes[1,0].set_xlabel('Car Age (Years)')
            axes[1,0].set_ylabel('Price (Yuan)')
            axes[1,0].set_title('Car Age vs Price')
            
            # Average price by registration year
            yearly_avg_price = self.df.groupby('reg_year')['price'].mean()
            yearly_avg_price.plot(kind='line', ax=axes[1,1], marker='o')
            axes[1,1].set_title('Average Price Trend by Registration Year')
            axes[1,1].set_xlabel('Registration Year')
            axes[1,1].set_ylabel('Average Price (Yuan)')
            
            plt.tight_layout()
            plt.savefig('time_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def anonymous_features_analysis(self):
        """Anonymous features analysis"""
        print("\n" + "="*50)
        print("7️⃣  Anonymous Features Analysis")
        print("="*50)
        
        # Get anonymous features
        anonymous_features = [col for col in self.df.columns if col.startswith('v_')]
        print(f"📊 Number of anonymous features: {len(anonymous_features)}")
        
        if anonymous_features:
            # Anonymous features basic statistics
            print("\n📈 Anonymous features descriptive statistics:")
            print(self.df[anonymous_features].describe())
            
            # Anonymous features correlation with price
            print(f"\n💰 Anonymous features correlation with price:")
            v_correlations = []
            for feature in anonymous_features:
                corr = self.df[feature].corr(self.df['price'])
                v_correlations.append((feature, corr))
                print(f"{feature:>6}: {corr:7.3f}")
            
            # Visualize anonymous features correlation
            v_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            features, corr_values = zip(*v_correlations)
            
            plt.figure(figsize=(12, 8))
            colors = ['red' if x < 0 else 'blue' for x in corr_values]
            plt.barh(features, corr_values, color=colors, alpha=0.7)
            plt.title('Anonymous Features-Price Correlation Coefficients')
            plt.xlabel('Correlation Coefficient')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig('anonymous_features_correlation.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Anonymous features correlation heatmap
            plt.figure(figsize=(12, 10))
            corr_matrix = self.df[anonymous_features + ['price']].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Anonymous Features Correlation Heatmap')
            plt.tight_layout()
            plt.savefig('anonymous_features_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Find top anonymous features with highest price correlation
            top_v_features = sorted(v_correlations, key=lambda x: abs(x[1]), reverse=True)[:5]
            print(f"\n🔝 Top 5 anonymous features with highest price correlation:")
            for feature, corr in top_v_features:
                print(f"{feature}: {corr:.3f}")
    
    def outlier_analysis(self):
        """Outlier analysis"""
        print("\n" + "="*50)
        print("8️⃣  Outlier Analysis")
        print("="*50)
        
        # Price outlier analysis
        print("💰 Price outlier analysis:")
        price = self.df['price']
        Q1 = price.quantile(0.25)
        Q3 = price.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        price_outliers = self.df[(price < lower_bound) | (price > upper_bound)]
        print(f"Normal price range: [{lower_bound:.0f}, {upper_bound:.0f}]")
        print(f"Price outliers count: {len(price_outliers)} ({len(price_outliers)/len(self.df)*100:.2f}%)")
        
        if len(price_outliers) > 0:
            print(f"Abnormally high-priced cars (top 5):")
            high_price_outliers = price_outliers.nlargest(5, 'price')[['SaleID', 'price', 'brand', 'model', 'power', 'kilometer']]
            print(high_price_outliers)
            
            print(f"\nAbnormally low-priced cars (top 5):")
            low_price_outliers = price_outliers.nsmallest(5, 'price')[['SaleID', 'price', 'brand', 'model', 'power', 'kilometer']]
            print(low_price_outliers)
        
        # Other numerical features outliers
        numerical_features = ['power', 'kilometer']
        existing_features = [f for f in numerical_features if f in self.df.columns]
        
        for feature in existing_features:
            print(f"\n🔍 {feature} outlier analysis:")
            
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)]
            
            print(f"Normal range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"Outliers count: {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)")
        
        # Visualize outliers
        features_to_plot = ['price'] + existing_features
        fig, axes = plt.subplots(1, len(features_to_plot), figsize=(6*len(features_to_plot), 5))
        if len(features_to_plot) == 1:
            axes = [axes]
        
        for i, feature in enumerate(features_to_plot):
            axes[i].boxplot(self.df[feature])
            axes[i].set_title(f'{feature} Box Plot')
            axes[i].set_ylabel(feature)
        
        plt.tight_layout()
        plt.savefig('outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_engineering_suggestions(self):
        """Feature engineering suggestions"""
        print("\n" + "="*50)
        print("9️⃣  Feature Engineering Suggestions")
        print("="*50)
        
        suggestions = [
            "🔧 Feature engineering suggestions based on EDA analysis:",
            "",
            "1. Target variable processing:",
            "   - Price distribution is right-skewed, consider log transformation: log(price+1)",
            "   - Handle price outliers using truncation or removal strategy",
            "",
            "2. Time features:",
            "   - Car age feature created: car_age = current_year - reg_year",
            "   - Create usage intensity: usage_intensity = kilometer / car_age",
            "   - Extract seasonal features from registration month",
            "",
            "3. Categorical feature encoding:",
            "   - Use target encoding for high-cardinality categorical variables (like brand)",
            "   - Use one-hot encoding for low-cardinality categorical variables",
            "   - Consider brand tier grouping (based on average price)",
            "",
            "4. Numerical feature transformation:",
            "   - Power and kilometer may need standardization",
            "   - Create power density related features",
            "   - Binning for kilometer",
            "",
            "5. Combined features:",
            "   - Value-for-money indicator: power / price",
            "   - Comprehensive car condition score: combine mileage, age, damage status",
            "   - Brand-model combination features",
            "",
            "6. Anonymous features:",
            "   - v_0 to v_14 are already important features, use directly",
            "   - Try feature selection to find most important anonymous features",
            "   - Consider PCA dimensionality reduction",
            "",
            "7. Missing value handling:",
            "   - Treat '-' in notRepairedDamage as independent category",
            "   - Fill numerical missing values with median",
            "",
            "8. Model suggestions:",
            "   - Recommend tree models: XGBoost, LightGBM, CatBoost",
            "   - Evaluation metrics: MAE, RMSE, MAPE",
            "   - Cross-validation strategy: 5-fold or 10-fold"
        ]
        
        for suggestion in suggestions:
            print(suggestion)
    
    def generate_summary_report(self):
        """Generate summary report"""
        print("\n" + "="*60)
        print("📋 Training Data EDA Summary Report")
        print("="*60)
        
        # Basic statistics
        price_stats = self.df['price'].describe()
        
        report = []
        report.append("# Used Car Training Dataset EDA Summary Report\n")
        report.append(f"## Data Overview")
        report.append(f"- Data size: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        report.append(f"- Target variable: price (used car price)")
        report.append(f"- Feature types: basic info + vehicle attributes + anonymous features")
        
        report.append(f"\n## Price Analysis (Target Variable)")
        report.append(f"- Average price: {price_stats['mean']:,.0f} Yuan")
        report.append(f"- Median price: {price_stats['50%']:,.0f} Yuan")
        report.append(f"- Price range: {price_stats['min']:,.0f} - {price_stats['max']:,.0f} Yuan")
        report.append(f"- Standard deviation: {price_stats['std']:,.0f} Yuan")
        
        # Correlation analysis
        if hasattr(self, 'df') and 'power' in self.df.columns:
            power_corr = self.df['power'].corr(self.df['price'])
            report.append(f"\n## Key Findings")
            report.append(f"- Power vs price correlation: {power_corr:.3f}")
            
        if 'car_age' in self.df.columns:
            age_corr = self.df['car_age'].corr(self.df['price'])
            report.append(f"- Car age vs price correlation: {age_corr:.3f}")
        
        report.append(f"\n## Data Quality")
        missing_count = self.df.isnull().sum().sum()
        report.append(f"- Standard missing values: {missing_count}")
        report.append(f"- Special values: notRepairedDamage field contains '-'")
        
        report.append(f"\n## Modeling Suggestions")
        report.append(f"1. Price distribution is right-skewed, suggest log transformation")
        report.append(f"2. Focus on anonymous features v_0 to v_14")
        report.append(f"3. Create car age, usage intensity and other derived features")
        report.append(f"4. Recommend tree models (XGBoost/LightGBM)")
        
        # Save report
        with open('train_eda_summary_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("💾 Training data EDA summary report saved to 'train_eda_summary_report.md'")
        
        for line in report:
            print(line)
    
    def run_full_eda(self):
        """Run complete EDA analysis"""
        if self.df is None:
            print("❌ Data not loaded, cannot perform analysis")
            return
        
        try:
            self.basic_info()
            self.target_variable_analysis()  # Focus on target variable
            self.missing_values_analysis()
            self.feature_price_relationship()  # Feature-price relationships
            self.categorical_analysis()
            self.time_analysis()
            self.anonymous_features_analysis()  # Anonymous features analysis
            self.outlier_analysis()
            self.feature_engineering_suggestions()
            self.generate_summary_report()
            
            print("\n" + "="*60)
            print("✅ Training data EDA analysis completed!")
            print("📁 Generated files:")
            print("   - train_eda_summary_report.md (summary report)")
            print("   - price_distribution_analysis.png (price distribution)")
            print("   - feature_price_correlation.png (feature correlation)")
            print("   - anonymous_features_correlation.png (anonymous features)")
            print("   - Other analysis charts...")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error during EDA analysis: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    # Create EDA analysis object
    eda = TrainDataEDA('used_car_train_20200313.csv')
    
    # Run complete analysis
    eda.run_full_eda()

if __name__ == "__main__":
    main()
