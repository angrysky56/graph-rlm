---
title: "5 Useful Python Scripts for Effective Feature Engineering"
source: "https://www.kdnuggets.com/5-useful-python-scripts-for-effective-feature-engineering"
author:
  - "[[Bala Priya C]]"
published:
created: 2026-01-18
description: "Feature engineering doesn’t have to be complex. These 5 Python scripts help you create meaningful features that improve model performance."
tags:
  - "clippings"
---
Feature engineering doesn’t have to be complex. These 5 Python scripts help you create meaningful features that improve model performance.

By , KDnuggets Contributing Editor & Technical Content Specialist on January 13, 2026 in

---

  

![Useful Python Scripts for Effective Feature Engineering](https://www.kdnuggets.com/wp-content/uploads/bala-python-feature-engineering-scripts.jpeg)  
Image by Author

## \# Introduction

  
As a machine learning practitioner, you know that feature engineering is painstaking, manual work. You need to create interaction terms between features, encode categorical variables properly, extract temporal patterns from dates, generate aggregations, and transform distributions. For each potential feature, you test whether it improves model performance, iterate on variations, and track what you've tried.[![NWU - Choose from a wide range of AI courses.](https://www.kdnuggets.com/wp-content/uploads/s-nwu-2512.jpg)  
Choose from a wide range of AI courses.](https://sps.northwestern.edu/information/data-science-online-masters.html?utm_source=kdnuggets&utm_medium=banner300x250&utm_campaign=kdnuggets_msds_banner300x250_l&utm_term=sep25&utm_content=msds&src=kdnuggets_msds_banner300x250_sepfy26_l)

This becomes more challenging as your dataset grows. With dozens of features, you will need systematic approaches to generate candidate features, evaluate their usefulness, and select the best ones. Without automation, you will likely miss valuable feature combinations that could significantly boost your model's performance.[![NVIDIA RTX PRO 6000 Blackwell Server Edition](https://www.kdnuggets.com/wp-content/uploads/s-pny-0126.jpg)  
Explore features](https://www.pny.com/nvidia-rtx-pro-6000-blackwell?iscommercial=true&utm_source=KDNuggets+Banner+300x250&utm_medium=Web+Banners&utm_campaign=Blackwell+Server&utm_id=RTX+PRO+6000)

This article covers five Python scripts specifically designed to automate the most impactful feature engineering tasks. These scripts help you generate high-quality features systematically, evaluate them objectively, and build optimized feature sets that maximize model performance.

**[You can find the code on GitHub](https://github.com/balapriyac/data-science-tutorials/tree/main/useful-feature-engineering-python-scripts)**.

## \# 1. Encoding Categorical Features

#### // The Pain Point

Categorical variables are everywhere in real-world data. You need to encode these categories, and choosing the right encoding method matters:

- One-hot encoding works for low-cardinality features but creates dimensionality problems with high-cardinality categories
- Label encoding is memory-efficient but implies ordinality
- Target encoding is powerful but risks data leakage

Implementing these encodings correctly, handling unseen categories in test data, and maintaining consistency across train, validation, and test splits require careful, error-prone code.

#### // What The Script Does

The script automatically selects and applies appropriate encoding strategies based on feature characteristics: cardinality, target correlation, and data type.

It handles one-hot encoding for low-cardinality features, target encoding for features correlated with the target, frequency encoding for high-cardinality features, and label encoding for ordinal variables. It also groups rare categories automatically, handles unseen categories in test data gracefully, and maintains encoding consistency across all data splits.

#### // How It Works

The script analyzes each categorical feature to determine its cardinality and relationship with the target variable.

- For features with fewer than 10 unique values, it applies one-hot encoding
- For high-cardinality features with more than 50 unique values, it uses frequency encoding to avoid dimensionality explosion
- For features showing correlation with the target, it applies target encoding with smoothing to prevent overfitting
- Rare categories appearing in less than 1% of rows are grouped into an "other" category

All encoding mappings are stored and can be applied consistently to new data, with unseen categories handled by defaulting to a rare category encoding or global mean.

⏩ **[Get the categorical feature encoder script](https://github.com/balapriyac/data-science-tutorials/blob/main/useful-feature-engineering-python-scripts/smart_encoder.py)**

## \# 2. Transforming Numerical Features

#### // The Pain Point

Raw numeric features often need transformation before modeling. Skewed distributions should be normalized, outliers should be handled, features with different scales need standardization, and non-linear relationships might require polynomial or logarithmic transformations. Manually testing different transformation strategies for each numeric feature is tedious. This process needs to be repeated for every numeric column and validated to ensure you are actually improving model performance.

#### // What The Script Does

The script automatically tests multiple transformation strategies for numeric features: log transforms, **[Box-Cox transformations](https://www.statisticshowto.com/probability-and-statistics/normal-distributions/box-cox-transformation/)**, square root, cube root, standardization, normalization, robust scaling, and power transforms.

It evaluates each transformation's impact on distribution normality and model performance, selects the best transformation for each feature, and applies transformations consistently to train and test data. It also handles zeros and negative values appropriately, avoiding transformation errors.

#### // How It Works

For each numeric feature, the script tests multiple transformations and evaluates them using normality tests — such as **[Shapiro-Wilk](https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test)** and **[Anderson-Darling](https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test)** — and distribution metrics like skewness and kurtosis. For features with skewness greater than 1, it prioritizes log and Box-Cox transformations.

For features with outliers, it applies robust scaling. The script maintains transformation parameters fitted on training data and applies them consistently to validation and test sets. Features with negative values or zeros are handled with shifted transformations or **[Yeo-Johnson](https://feature-engine.trainindata.com/en/1.8.x/user_guide/transformation/YeoJohnsonTransformer.html)** transformations that work with any real values.

⏩ **[Get the numerical feature transformer script](https://github.com/balapriyac/data-science-tutorials/blob/main/useful-feature-engineering-python-scripts/numerical_transformer.py)**

## \# 3. Generating Feature Interactions

#### // The Pain Point

Interactions between features often contain valuable signal that individual features miss. Revenue might matter differently across customer segments, advertising spend might have different effects by season, or the combination of product price and category might be more predictive than either alone. But with dozens of features, testing all possible pairwise interactions means evaluating thousands of candidates.

#### // What The Script Does

This script generates feature interactions using mathematical operations, polynomial features, ratio features, and categorical combinations. It evaluates each candidate interaction's predictive power using mutual information or model-based importance scores. It returns only the top N most valuable interactions, avoiding feature explosion while capturing the most impactful combinations. It also supports custom interaction functions for domain-specific feature engineering.

#### // How It Works

The script generates candidate interactions between all feature pairs:

- For numeric features, it creates products, ratios, sums, and differences
- For categorical features, it creates joint encodings

Each candidate is scored using mutual information with the target or feature importance from a random forest. Only interactions exceeding an importance threshold or ranking in the top N are retained. The script handles edge cases like division by zero, infinite values, and correlations between generated features and original features. Results include clear feature names showing which original features were combined and how.

⏩ **[Get the feature interaction generator script](https://github.com/balapriyac/data-science-tutorials/blob/main/useful-feature-engineering-python-scripts/interaction_generator.py)**

## \# 4. Extracting Datetime Features

#### // The Pain Point

Datetime columns contain useful temporal information, but using them effectively requires extensive manual feature engineering. You need to do the following:

- Extract components like year, month, day, and hour
- Create derived features such as day of week, quarter, and weekend flags
- Compute time differences like days since a reference date and time between events
- Handle cyclical patterns

Writing this extraction code for every datetime column is repetitive and time-consuming, and practitioners often forget valuable temporal features that could improve their models.

#### // What The Script Does

The script automatically extracts comprehensive datetime features from timestamp columns, including basic components, calendar features, boolean indicators, cyclical encodings using sine and cosine transformations, season indicators, and time differences from reference dates. It also detects and flags holidays, handles multiple datetime columns, and computes time differences between datetime pairs.

#### // How It Works

The script takes datetime columns and systematically extracts all relevant temporal patterns.

For cyclical features like month or hour, it creates sine and cosine transformations:  

This ensures that December and January are close in the feature space. It calculates time deltas from a reference point (days since epoch, days since a specific date) to capture trends.

For datasets with multiple datetime columns (e.g. `order_date` and `ship_date`), it computes differences between them to find durations like `processing_time`. Boolean flags are created for special days, weekends, and period boundaries. All features use clear naming conventions showing their source and meaning.

⏩ **[Get the datetime feature extractor script](https://github.com/balapriyac/data-science-tutorials/blob/main/useful-feature-engineering-python-scripts/datetime_extractor.py)**

## \# 5. Selecting Features Automatically

#### // The Pain Point

After feature engineering, you usually have several features, many of which are redundant, irrelevant, or cause overfitting. You need to identify which features actually help your model and which ones should be removed. Manual feature selection means training models repeatedly with different feature subsets, tracking results in spreadsheets, and trying to understand complex feature importance scores. The process is slow and subjective, and you never know if you have found the optimal feature set or just got lucky with your trials.

#### // What The Script Does

The script automatically selects the most valuable features using multiple selection methods:

- Variance-based filtering removes constant or near-constant features
- Correlation-based filtering removes redundant features
- Statistical tests like analysis of variance (ANOVA), chi-square, and mutual information
- Tree-based feature importance
- L1 regularization
- Recursive feature elimination

It then combines results from multiple methods into an ensemble score, ranks all features by importance, and identifies the optimal feature subset that maximizes model performance while minimizing dimensionality.

#### // How It Works

The script applies a multi-stage selection pipeline. Here is what each stage does:

1. Remove features with zero or near-zero variance as they provide no information
2. Remove highly correlated feature pairs, keeping the one more correlated with the target
3. Calculate feature importance using multiple methods, such as random forest importance, mutual information scores, statistical tests, and L1 regularization coefficients
4. Normalize and combine scores from different methods into an ensemble ranking
5. Use recursive feature elimination or cross-validation to determine the optimal number of features

The result is a ranked list of features and a recommended subset for model training, along with detailed importance scores from each method.

⏩ **[Get the automated feature selector script](https://github.com/balapriyac/data-science-tutorials/blob/main/useful-feature-engineering-python-scripts/feature_selector.py)**

## \# Conclusion

  
These five scripts address the core challenges of feature engineering that consume the majority of time in machine learning projects. Here is a quick recap:

- Categorical encoder handles encoding intelligently based on cardinality and target correlation
- Numerical transformer automatically finds optimal transformations for each numeric feature
- Interaction generator discovers valuable feature combinations systematically
- Datetime extractor extracts comprehensive temporal patterns and cyclical features
- Feature selector identifies the most predictive features using ensemble methods

Each script can be used independently for specific feature engineering tasks or combined into a complete pipeline. Start with the encoders and transformers to prepare your base features, use the interaction generator to discover complex patterns, extract temporal features from datetime columns, and finish with feature selection to optimize your feature set.

Happy feature engineering!  
  

is a developer and technical writer from India. She likes working at the intersection of math, programming, data science, and content creation. Her areas of interest and expertise include DevOps, data science, and natural language processing. She enjoys reading, writing, coding, and coffee! Currently, she's working on learning and sharing her knowledge with the developer community by authoring tutorials, how-to guides, opinion pieces, and more. Bala also creates engaging resource overviews and coding tutorials.

  

---

  

[<= Previous post](https://www.kdnuggets.com/we-tried-5-missing-data-imputation-methods-the-simplest-method-won-sort-of)

[Next post =>](https://www.kdnuggets.com/csv-vs-parquet-vs-arrow-storage-formats-explained)