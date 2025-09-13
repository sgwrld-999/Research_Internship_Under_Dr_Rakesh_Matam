# Random Forest Experiments - Results Comparison

This document provides a comparison of different Random Forest configurations tested on the network intrusion detection dataset.

## Experiment Summary

We ran three different configurations of Random Forest models, progressively improving the model's performance:

| Metric | Experiment 1 | Experiment 2 | Experiment 3 |
|--------|--------------|--------------|--------------|
| Test Accuracy | 20.00% | 91.33% | 92.27% |
| F1-Score (Macro) | 6.67% | 91.10% | 92.17% |
| F1-Score (Weighted) | 6.67% | 91.10% | 92.17% |
| Out-of-Bag Score | 19.46% | 91.34% | 92.30% |
| Training Time | ~4 seconds | ~43 seconds | ~107 seconds |

## Key Configuration Differences

### Experiment 1 (Baseline)
- 100 trees
- Maximum depth of 5 (very shallow)
- Gini impurity criterion
- High regularization values (min_weight_fraction_leaf: 0.3, min_impurity_decrease: 0.3)
- Basic "balanced" class weight

### Experiment 2 (Optimized)
- 500 trees
- Maximum depth of 20
- Entropy criterion
- Reduced regularization
- "balanced_subsample" class weight
- Bootstrap sampling with max_samples: 0.8

### Experiment 3 (Maximum Accuracy)
- 1000 trees
- Maximum depth of 25
- Entropy criterion
- Further reduced regularization
- 70% of features used for each split
- 70% of samples used for each tree

## Conclusions

1. **Experiment 1** performed at baseline level (20% accuracy for 5 classes), likely due to excessive regularization and insufficient model complexity.

2. **Experiment 2** achieved a dramatic improvement to 91.33% accuracy by:
   - Increasing model complexity
   - Using entropy for multi-class problems
   - Reducing overly strict regularization
   - Better handling of class imbalance

3. **Experiment 3** further improved to 92.27% accuracy by:
   - Doubling the number of trees
   - Increasing depth
   - Fine-tuning feature and sample usage
   - Optimizing regularization parameters

## Recommendations

For this network intrusion detection dataset, we recommend using the Experiment 3 configuration for production models, as it provides the best accuracy with acceptable training time. For faster inference with slightly lower accuracy, Experiment 2 configuration provides a good balance of performance and speed.

The most important parameters for achieving high accuracy on this dataset were:
1. Number of trees (more is better)
2. Tree depth (deeper trees capture more complex patterns)
3. Entropy criterion (better for multi-class problems)
4. Proper regularization (avoid over-regularization)
5. Using "balanced_subsample" for class weights