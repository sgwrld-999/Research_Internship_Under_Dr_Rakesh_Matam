# **Linformer-Based Intrusion Detection System: Metrics and Plots**

### **Overview and Testing Objectives**

The algorithm's purpose is to provide an **efficient and lightweight Intrusion Detection System (IDS)** using a Linformer architecture. The core innovation is its linear time complexity ($O(n)$), which aims to overcome the performance bottlenecks of standard Transformers ($O(n^2)$). A key challenge in modern IoT and Consumer Electronics (CE) environments is balancing "detection performance and resource consumption." This evaluation framework is designed to rigorously test this balance.

The primary testing objectives are:
1.  To evaluate the model's **classification accuracy** in identifying malicious network traffic for both binary (attack vs. normal) and multi-class (specific attack types) scenarios.
2.  To determine if the observed performance differences between the Linformer-IDS and other models are **statistically significant**.
3.  To quantify the algorithm's **efficiency** in terms of time and computational resources, validating its suitability for real-time and edge deployment.
4.  To analyze the model's behavior and misclassifications through detailed visualizations.

---
### **Evaluation of Classification Performance**

This section evaluates the model's ability to correctly classify network traffic.

#### **Binary Classification**

For the task of distinguishing between 'Normal' and 'Attack' traffic.

1.  **Accuracy**:
    * **What it measures**: The percentage of total predictions that were correct.
    * **Relevance**: Provides a simple, high-level overview of the model's overall correctness.
    * **Strengths/Limitations**: A good initial indicator, but can be highly misleading on imbalanced datasets. A model that only predicts the majority class can achieve high accuracy but fail at its primary goal of detecting attacks.

2.  **Precision**:
    * **What it measures**: Of all the instances the model predicted as an attack, the percentage that were *actual* attacks.
    * **Relevance**: Crucial for minimizing **False Positives**. High precision ensures that when an alarm is raised, it is trustworthy.

3.  **Recall (Sensitivity)**:
    * **What it measures**: Of all the *actual* attacks that occurred, the percentage that the model correctly identified.
    * **Relevance**: Vital for minimizing **False Negatives**. A false negative represents a missed attack—the most critical failure for an IDS.

4.  **F1-Score**:
    * **What it measures**: The harmonic mean of Precision and Recall.
    * **Relevance**: This is arguably the most important single metric for an imbalanced task like intrusion detection, as it provides a balanced measure of a model's performance.

#### **Multi-class Classification**

For the task of identifying specific attack categories.

1.  **Accuracy, Precision, Recall, F1-Score**: These metrics are still fundamental. They are calculated for each class and then aggregated.

2.  **Macro and Micro Averaged Metrics**:
    * **Macro Average**: Calculates the metric independently for each class and then averages them. This treats all classes equally, which is important for evaluating performance on rare but critical attack types.
    * **Micro Average**: Aggregates the contributions of all classes to compute the average metric, which is effectively the overall accuracy.

3.  **Confusion Matrix**:
    * **What it shows**: A matrix providing a detailed breakdown of correct and incorrect predictions for each class.
    * **Relevance**: Essential for understanding *how* the model is failing.

4.  **ROC-AUC for Each Class (One-vs-Rest)**:
    * **What it measures**: The Area Under the Receiver Operating Characteristic Curve. It measures the model's ability to distinguish between classes.
    * **Relevance**: An AUC of 1.0 represents a perfect classifier, while 0.5 represents a random guess. It is a robust metric against shifts in class distribution.

---
### **Evaluation of Statistical Significance**

To ensure that observed differences in performance metrics (e.g., the Linformer's F1-score vs. a standard Transformer's) are not due to random chance, statistical hypothesis tests must be performed.

* **Pre-Assumptions (Hypotheses)**: Before testing, we establish two competing hypotheses:
    * **Null Hypothesis (H₀)**: There is *no significant difference* in performance between the two models being compared. The observed difference is a result of statistical noise or random sampling of the test data.
    * **Alternative Hypothesis (H₁)**: There *is a significant difference* in performance. The superiority of one model over another is statistically meaningful.

* **Proposed Tests**:
    1.  **Paired t-test**:
        * **What it measures**: Compares the means of two related groups to determine if there is a statistically significant difference between them.
        * **Relevance**: Ideal for comparing the performance scores (e.g., accuracy from 10-fold cross-validation) of two models (e.g., Linformer vs. CNN) on the same dataset splits. It helps confirm if one model is consistently better than another.

    2.  **ANOVA (Analysis of Variance)**:
        * **What it measures**: Compares the means of three or more groups.
        * **Relevance**: Use this to compare the performance of the Linformer-IDS against multiple other algorithms (e.g., CNN, LSTM, Deep Forest) simultaneously to see if there is any significant difference within the group of models.

* **Interpreting Results (p-value)**: The tests produce a **p-value**. This value is the probability of observing the results if the null hypothesis were true. A commonly used significance level is 0.05.
    * If **p < 0.05**: We **reject the null hypothesis**. The observed performance difference is statistically significant.
    * If **p ≥ 0.05**: We **fail to reject the null hypothesis**. We do not have enough evidence to say the difference is real.

---
### **Evaluation of Parameter Selection**

* **What it measures**: The complexity and sensitivity of the model to its hyperparameter settings.
* **Relevance**: Models that require extensive, dataset-specific tuning are harder to deploy and maintain. The evaluation should analyze how well a simple configuration performs across different datasets.
* **Strengths/Limitations reflected**: A model that performs well without extensive tuning is more robust and practical. The Linformer's primary hyperparameter is the projected dimension `k`, and its performance sensitivity to this value should be analyzed.

---
### **Evaluation of Algorithm Time Consumption**

* **What it measures**: The time required for the model to process data, specifically the average inference time per flow.
* **Relevance**: This is a critical metric for a real-time IDS. Low inference time is mandatory to keep up with high-volume network traffic without becoming a bottleneck.
* **Strengths/Limitations reflected**: This directly tests the core hypothesis of the Linformer. Its performance should be significantly faster than standard deep learning models like CNN and LSTM.

---
### **Evaluation of Algorithm Resource Consumption**

* **What it measures**:
    1.  **Memory Usage**: Peak operational memory (RAM/VRAM) consumed during inference.
    2.  **Model Size**: The amount of flash memory or disk space required to store the trained model.
    3.  **CPU Usage**: The percentage of CPU utilized during inference.
* **Relevance**: This is paramount for deployment on resource-constrained CE and IoT devices.
* **Strengths/Limitations reflected**: A small model size and low memory/CPU usage would validate the Linformer-IDS as a practical solution for edge computing.

---
### **Plots for Both Binary and Multi-class**

1.  **ROC Curve**:
    * **Purpose**: To visualize the trade-off between the true positive rate and the false positive rate.
    * **Interpretation**: A curve that bows towards the top-left corner indicates a better-performing model.

2.  **Confusion Matrix**:
    * **Purpose**: To provide a detailed, visual breakdown of classification results.
    * **Interpretation**: A perfect model would have a bright, solid diagonal line. Off-diagonal values highlight misclassifications.

3.  **Precision-Recall Curve**:
    * **Purpose**: To visualize the trade-off between precision and recall for different thresholds.
    * **Interpretation**: This is especially useful for imbalanced datasets. A curve that stays high and close to the top-right corner indicates that the model can achieve high recall without sacrificing much precision.