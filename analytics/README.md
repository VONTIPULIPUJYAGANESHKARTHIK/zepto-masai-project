# Module 2: Analytics Pipeline (Titanic)

This module handles the end-to-end data science workflow on the classic Titanic dataset. The pipeline is split into two ordered notebooks that share a single offline `titanic.csv` generated on the first run.

## 1. Missing Value Strategies
- **`deck` (77.2% missing)**: Missing rate is >30%. Imputation would be completely unreliable. Dropped the column.
- **`age` (19.9% missing)**: Missing rate is between 5% and 30%. Imputed with the median age.
- **`embarked` / `embark_town` (0.22% missing)**: Missing rate is <5%. Dropped those specific rows.

## 2. Fare Skewness
Since Mean (32.10) > Median (14.45) > Mode (8.05), the `fare` distribution is strongly right-skewed (positive skewness).

## 3. Correlation Interpretations
- **`pclass` and `fare` (-0.55)**: The strongest off-diagonal correlation. Heavily negative, indicating that lower passenger class numbers (1st class) paid significantly higher fares.
- **`pclass` and `survived` (-0.34)**: The second strongest. Negative, meaning as class number increases (e.g., 3rd class), survival likelihood decreases, highlighting the priority given to upper-class passengers for lifeboats.

## 4. Multivariate Data Story (Chart Interpretations)
1. **Survival by Sex and Class**: The bar chart demonstrates the "women and children first" protocol compounded by class privilege. Females in 1st/2nd class had near-perfect survival (>90%), while 3rd-class males had the absolute worst outcomes.
2. **Age and Sex Distribution by Survival**: The violin plot reveals a distinct bulge for male survivors representing young boys. Male children were saved while adult males were left behind. 
3. **Fare vs Age Colored by Survival**: The scatterplot (log scale) shows survivors heavily clustered in the upper half (higher fares/1st class). Victims cluster at the bottom across all ages, proving cheapest-fare passengers were disproportionately abandoned.
4. **Survival by Port and Class**: The heatmap shows 1st-class passengers boarding at Cherbourg had an exceptionally high survival rate (~70%), while 3rd-class from Southampton had under 20%. Port of embarkation is heavily confounded with passenger class.

## 5. Preprocessing & Split Strategy
A Stratified Train/Test split was used to preserve the natural survival imbalance (~38% survival). All preprocessing (Imputation, OneHotEncoding, StandardScaler) was strictly built inside a `ColumnTransformer` and fit *only* on the training fold to prevent data leakage.

## 6. Imbalance Strategy Comparison
Baseline logistic regression had high precision but lower recall. Applying `class_weight='balanced'` and `SMOTE` prioritized capturing actual survivors (Recall increased), but at the cost of more false positives (Precision dropped). `class_weight` performed nearly identical to SMOTE for this dataset.

## 7. Heteroscedasticity in Fare Regression
The residual plot for predicting `fare` displays a clear "funnel" shape. This is textbook heteroscedasticity, meaning our linear regression struggles as the predicted fare increases, likely because true fares are heavily right-skewed and non-linear.

## 8. Final Model Comparison Table

### Classifiers (Predicting `survived`)
| Metric | Logistic Regression | Decision Tree | Tuned Random Forest |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.815 | 0.798 | 0.803 |
| **Precision** | 0.776 | 0.763 | 0.811 |
| **Recall** | 0.725 | 0.696 | 0.623 |
| **F1 Score** | 0.750 | 0.728 | 0.704 |
| **AUC** | 0.865 | 0.835 | 0.871 |

### Regression Side-Task (Predicting `fare`)
| Metric | Linear Regression |
| :--- | :--- |
| **MAE** | 23.35 |
| **RMSE** | 42.13 |
| **R2** | 0.328 |
| **Adj R2** | 0.308 |

*Note: Classifier metrics and Regression metrics are on different scales and cannot be directly compared.*

## 9. Final Deployment Recommendation
I recommend deploying the **Tuned Random Forest**. It achieved the highest overall AUC (0.871) and the best precision (0.811). While recall is slightly lower than the baseline logistic regression, its ability to perfectly capture non-linear relationships in complex features makes it the most robust choice for future unseen data. The complete pipeline (including encoders and scalers) has been saved to `model_pipeline.joblib`.
