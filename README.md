# Human Activity Recognition (HAR): Edge-Device Optimization Pipeline

## Overview
This project implements a highly optimized, supervised Machine Learning pipeline to classify human physical activities (walking, sitting, standing, etc.) from smartphone sensor data. 

While the baseline dataset contains 561 high-dimensional features from embedded accelerometers and gyroscopes, deploying such a heavy model to a memory-constrained edge device (like a smartwatch) drains battery life and introduces severe latency. 

This project explores, compares, and engineers multiple dimensionality reduction techniques—specifically contrasting **Principal Component Analysis (PCA)** against an academic **K-Means Feature Selection** strategy—to identify the ultimate lightweight deployment model.

## Methodology & Engineering Pipeline

### 1. Data Preprocessing & Leakage Prevention
* **Data Extraction:** Raw sensor signals were extracted and combined from the UCI Machine Learning Repository.
* **Leakage Prevention:** To ensure mathematical integrity, the data was rigorously split into training and testing sets *before* any scaling occurred.
* **Standardization:** A `StandardScaler` was applied to normalize the sensor features, preventing high-magnitude variables from dominating the mathematical algorithms.

### 2. Dimensionality Reduction Showdown
To reduce the 561-feature footprint, two distinct reduction strategies were tested against a Logistic Regression and a Random Forest classifier:

* **Strategy A: PCA (Feature Extraction)**
  Using Cumulative Explained Variance (the Elbow Method), the algorithm calculated that preserving 95% of the dataset's variance required exactly 101 components. 
  * *Result:* While Logistic Regression performed well on PCA data, the Random Forest model suffered a massive performance drop (down to 93.68%). Decision Trees struggle to process the dense, blended mathematical "blobs" created by PCA.

* **Strategy B: K-Means (Feature Selection)**
  Instead of mathematically blending features, K-Means was deployed horizontally across the transposed dataset to cluster highly correlated *sensors*. By selecting exactly one representative sensor from 100 distinct clusters, the pipeline successfully eliminated redundant noise.
  * *Result:* Because K-Means preserves pure, isolated, real-world sensor readings, Random Forest accuracy skyrocketed back up to 97.21%. 

## Final Results & Optimization Metrics

By evaluating the reduced models against the massive 561-feature baselines, the K-Means feature selection strategy proved to be the ultimate engineering solution.

| Model & Strategy | Features Used | Accuracy | Inference Time |
| :--- | :--- | :--- | :--- |
| **Baseline: Logistic Regression** | All 561 | 98.36% | ~ 5.60 s |
| **Baseline: Random Forest** | All 561 | 97.75% | ~ 16.41 s |
| | | | |
| **Optimized: LR + K-Means** | **100** | **96.13%** | **~ 1.39 s** |
| **Optimized: RF + K-Means** | 100 | 97.21% | ~ 4.08 s |

### The Deployment Verdict: Logistic Regression + K-Means
While Random Forest achieved a slightly higher raw accuracy, **Logistic Regression + K-Means** is the undisputed champion for edge-device deployment. 

1. **Hardware Efficiency:** By relying on K-Means feature selection, the smartphone app can physically ignore **82.1%** of the incoming sensor data, saving massive amounts of battery and memory.
2. **Speed:** The model inference time was reduced by **~400%** compared to the baseline, processing the data in a blistering 1.39 seconds.
3. **Accuracy Trade-off:** The model maintained a robust 96.13% accuracy, trading only a microscopic 2.2% drop from the heavy baseline to achieve these massive speed and hardware upgrades.

## Technical Stack
* **Language:** Python
* **Deployment Packaging:** `joblib`
* **Libraries:** Scikit-Learn (PCA, KMeans, StandardScaler, LogisticRegression, RandomForestClassifier), Pandas, NumPy, Matplotlib
