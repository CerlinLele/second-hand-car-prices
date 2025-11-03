# Used Car Training Dataset EDA Summary Report

## Data Overview
- Data size: 150,000 rows × 35 columns
- Target variable: price (used car price)
- Feature types: basic info + vehicle attributes + anonymous features

## Price Analysis (Target Variable)
- Average price: 5,923 Yuan
- Median price: 3,250 Yuan
- Price range: 11 - 99,999 Yuan
- Standard deviation: 7,502 Yuan

## Key Findings
- Power vs price correlation: 0.220
- Car age vs price correlation: -0.612

## Data Quality
- Standard missing values: 19168
- Special values: notRepairedDamage field contains '-'

## Modeling Suggestions
1. Price distribution is right-skewed, suggest log transformation
2. Focus on anonymous features v_0 to v_14
3. Create car age, usage intensity and other derived features
4. Recommend tree models (XGBoost/LightGBM)