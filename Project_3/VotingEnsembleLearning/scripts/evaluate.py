"""
Voting Ensemble Model Evaluation Script

This module provides comprehensive evaluation capabilities for trained Voting Ensemble models,
including performance metrics, visualizations, and detailed analysis.

THEORY - Voting Ensemble Evaluation Framework:
==============================================

Comprehensive Voting Ensemble evaluation involves multiple aspects:

1. ENSEMBLE PERFORMANCE METRICS:
   - Overall ensemble accuracy and standard metrics
   - Individual base learner performance analysis
   - Voting agreement and disagreement analysis
   - Ensemble diversity measures

2. VOTING MECHANISM ANALYSIS:
   - Hard voting: Majority class vote
   - Soft voting: Average of predicted probabilities
   - Voting confidence and uncertainty analysis
   - Decision boundary comparison

3. BASE LEARNER ANALYSIS:
   - Individual classifier performance
   - Feature importance from each base learner
   - Prediction correlation between base learners
   - Ensemble member contribution analysis

4. ENSEMBLE CHARACTERISTICS:
   - Diversity-accuracy trade-off
   - Error correlation analysis
   - Prediction agreement patterns
   - Confidence distribution analysis

THEORY - Voting Ensemble Specific Metrics:
==========================================

1. ENSEMBLE DIVERSITY MEASURES:
   - Disagreement Measure: Fraction of instances where classifiers disagree
   - Double-Fault Measure: Fraction where both classifiers are wrong
   - Q-Statistic: Measure of correlation between classifier errors
   - Kappa Statistic: Agreement measure beyond chance

2. VOTING ANALYSIS:
   - Vote Distribution: How votes are distributed across classes
   - Confidence Scores: Ensemble prediction confidence
   - Unanimous vs. Majority Decisions
   - Tie-breaking Analysis

3. BASE LEARNER CONTRIBUTION:
   - Individual Accuracy: Performance of each base learner
   - Marginal Contribution: Performance gain from each classifier
   - Feature Importance Aggregation across learners

Author: AI Assistant
Date: September 2025
Version: 1.0.0
"""

# Standard imports
import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any, Union
from itertools import combinations

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, log_loss,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Custom imports
from voting_ensemble.config_loader import VotingEnsembleConfig
from voting_ensemble.voting_ensemble_with_softmax import VotingEnsembleWithSoftmax

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")


class VotingEnsembleEvaluator:
    """
    Comprehensive evaluation framework for Voting Ensemble models.
    
    This class provides detailed model evaluation including performance metrics,
    statistical analysis, visualization, and interpretability features specific
    to Voting Ensemble classifiers.
    
    Attributes:
        model (VotingEnsembleWithSoftmax): The trained model to evaluate
        config (VotingEnsembleConfig): Configuration object
        evaluation_results (Dict): Stored evaluation results
        
    Example:
        >>> evaluator = VotingEnsembleEvaluator()
        >>> evaluator.load_model("model.joblib")
        >>> results = evaluator.evaluate_on_dataset("test_data.csv")
        >>> evaluator.generate_report()
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path (Optional[str]): Path to trained model file
        """
        self.model = None
        self.config = None
        self.evaluation_results = {}
        
        if model_path:
            self.load_model(model_path)
            
        # Setup directories
        self.setup_directories()
        
        logger.info("Voting Ensemble Evaluator initialized")
        
    def setup_directories(self) -> None:
        """Create necessary directories for outputs."""
        directories = [
            Path("evaluation_outputs"),
            Path("evaluation_plots"),
            Path("evaluation_reports")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def load_model(self, model_path: str) -> None:
        """
        Load a trained Voting Ensemble model.
        
        Args:
            model_path (str): Path to the saved model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model loading fails
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            # Load model data
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                # New format with metadata
                self.model = VotingEnsembleWithSoftmax(model_data['config'])
                self.model.model = model_data['model']
                self.model.label_encoder = model_data['label_encoder']
                self.model.scaler = model_data['scaler']
                self.model.feature_names = model_data['feature_names']
                self.model.training_history = model_data['training_history']
                self.model.is_fitted = True
                self.config = model_data['config']
            else:
                # Legacy format
                self.model = model_data
                self.config = getattr(model_data, 'config', None)
                
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
            
    def load_and_prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load and prepare evaluation dataset.
        
        Args:
            data_path (str): Path to the dataset CSV file
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Features, targets, feature names
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        logger.info(f"Loading evaluation data from {data_path}")
        
        # Load data
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded: {data.shape}")
        
        # Prepare features and targets
        if self.config and hasattr(self.config, 'target_column'):
            target_column = self.config.target_column
        else:
            # Assume last column is target
            target_column = data.columns[-1]
            logger.warning(f"Using last column as target: {target_column}")
            
        if self.config and hasattr(self.config, 'feature_columns') and self.config.feature_columns:
            feature_columns = self.config.feature_columns
        else:
            feature_columns = [col for col in data.columns if col != target_column]
            
        X = data[feature_columns].values.astype(np.float32)
        y = data[target_column].values
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        return X, y, feature_columns
        
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray): Predicted probabilities
            
        Returns:
            Dict[str, float]: Basic metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Add log loss if possible
        try:
            y_true_encoded = self.model.label_encoder.transform(y_true)
            metrics['log_loss'] = log_loss(y_true_encoded, y_proba)
        except Exception as e:
            logger.warning(f"Could not calculate log loss: {e}")
            
        # Add multiclass AUC if possible
        try:
            y_true_encoded = self.model.label_encoder.transform(y_true)
            metrics['auc_ovr_macro'] = roc_auc_score(y_true_encoded, y_proba, 
                                                   multi_class='ovr', average='macro')
            metrics['auc_ovr_weighted'] = roc_auc_score(y_true_encoded, y_proba, 
                                                      multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            
        return metrics
        
    def analyze_base_learners(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze individual base learner performance.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            
        Returns:
            Dict[str, Any]: Base learner analysis results
        """
        logger.info("Analyzing individual base learner performance...")
        
        if not hasattr(self.model.model, 'estimators_'):
            logger.warning("Cannot analyze base learners - estimators not available")
            return {}
            
        base_learner_analysis = {}
        
        # Prepare data
        X_processed, y_processed = self.model._prepare_data(X, y)
        
        # Analyze each base learner
        estimator_names = [name for name, _ in self.model.model.estimators]
        estimators = [estimator for _, estimator in self.model.model.estimators]
        
        individual_performances = {}
        individual_predictions = {}
        
        for name, estimator in zip(estimator_names, estimators):
            try:
                # Get predictions from individual estimator
                y_pred_individual = estimator.predict(X_processed)
                y_proba_individual = estimator.predict_proba(X_processed) if hasattr(estimator, 'predict_proba') else None
                
                # Calculate metrics for individual estimator
                individual_accuracy = accuracy_score(y_processed, y_pred_individual)
                individual_f1 = f1_score(y_processed, y_pred_individual, average='weighted', zero_division=0)
                
                individual_performances[name] = {
                    'accuracy': individual_accuracy,
                    'f1_weighted': individual_f1,
                    'predictions': y_pred_individual,
                    'probabilities': y_proba_individual
                }
                
                individual_predictions[name] = y_pred_individual
                
                logger.info(f"{name} - Accuracy: {individual_accuracy:.4f}, F1: {individual_f1:.4f}")
                
            except Exception as e:
                logger.warning(f"Could not analyze {name}: {e}")
                
        base_learner_analysis['individual_performances'] = individual_performances
        base_learner_analysis['estimator_names'] = estimator_names
        
        # Calculate diversity measures if we have multiple estimators
        if len(individual_predictions) > 1:
            diversity_measures = self.calculate_diversity_measures(individual_predictions, y_processed)
            base_learner_analysis['diversity_measures'] = diversity_measures
            
        return base_learner_analysis
        
    def calculate_diversity_measures(self, individual_predictions: Dict[str, np.ndarray], 
                                   y_true: np.ndarray) -> Dict[str, Any]:
        """
        Calculate ensemble diversity measures.
        
        Args:
            individual_predictions (Dict[str, np.ndarray]): Predictions from each base learner
            y_true (np.ndarray): True labels
            
        Returns:
            Dict[str, Any]: Diversity measures
        """
        logger.info("Calculating ensemble diversity measures...")
        
        diversity_measures = {}
        estimator_names = list(individual_predictions.keys())
        
        # Pairwise diversity measures
        pairwise_measures = {}
        
        for name1, name2 in combinations(estimator_names, 2):
            pred1 = individual_predictions[name1]
            pred2 = individual_predictions[name2]
            
            # Disagreement measure
            disagreement = np.mean(pred1 != pred2)
            
            # Double-fault measure
            correct1 = pred1 == y_true
            correct2 = pred2 == y_true
            double_fault = np.mean(~correct1 & ~correct2)
            
            # Q-statistic
            n11 = np.sum(correct1 & correct2)
            n10 = np.sum(correct1 & ~correct2)
            n01 = np.sum(~correct1 & correct2)
            n00 = np.sum(~correct1 & ~correct2)
            
            q_statistic = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10) if (n11 * n00 + n01 * n10) > 0 else 0
            
            pairwise_measures[f"{name1}_vs_{name2}"] = {
                'disagreement': disagreement,
                'double_fault': double_fault,
                'q_statistic': q_statistic
            }
            
        diversity_measures['pairwise'] = pairwise_measures
        
        # Average diversity measures
        disagreements = [measures['disagreement'] for measures in pairwise_measures.values()]
        double_faults = [measures['double_fault'] for measures in pairwise_measures.values()]
        q_statistics = [measures['q_statistic'] for measures in pairwise_measures.values()]
        
        diversity_measures['average'] = {
            'mean_disagreement': np.mean(disagreements),
            'mean_double_fault': np.mean(double_faults),
            'mean_q_statistic': np.mean(q_statistics)
        }
        
        logger.info(f"Average disagreement: {diversity_measures['average']['mean_disagreement']:.4f}")
        logger.info(f"Average double-fault: {diversity_measures['average']['mean_double_fault']:.4f}")
        
        return diversity_measures
        
    def analyze_voting_patterns(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze voting patterns and agreement.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            
        Returns:
            Dict[str, Any]: Voting pattern analysis
        """
        logger.info("Analyzing voting patterns...")
        
        if not hasattr(self.model.model, 'estimators_'):
            logger.warning("Cannot analyze voting patterns - estimators not available")
            return {}
            
        # Prepare data
        X_processed, y_processed = self.model._prepare_data(X, y)
        
        # Get ensemble predictions
        ensemble_pred = self.model.predict(X)
        ensemble_proba = self.model.predict_proba(X)
        
        # Get individual predictions
        individual_predictions = {}
        individual_probabilities = {}
        
        estimator_names = [name for name, _ in self.model.model.estimators]
        estimators = [estimator for _, estimator in self.model.model.estimators]
        
        for name, estimator in zip(estimator_names, estimators):
            try:
                individual_predictions[name] = estimator.predict(X_processed)
                if hasattr(estimator, 'predict_proba'):
                    individual_probabilities[name] = estimator.predict_proba(X_processed)
            except Exception as e:
                logger.warning(f"Could not get predictions from {name}: {e}")
                
        # Analyze voting agreement
        voting_analysis = {}
        
        # Calculate prediction agreement
        if len(individual_predictions) > 0:
            prediction_matrix = np.array(list(individual_predictions.values())).T
            
            # Unanimous decisions
            unanimous_mask = np.all(prediction_matrix == prediction_matrix[:, 0:1], axis=1)
            unanimous_rate = np.mean(unanimous_mask)
            
            # Majority decisions (for more than 2 estimators)
            if prediction_matrix.shape[1] > 2:
                majority_decisions = []
                for i in range(len(prediction_matrix)):
                    unique, counts = np.unique(prediction_matrix[i], return_counts=True)
                    max_count = np.max(counts)
                    is_majority = max_count > prediction_matrix.shape[1] // 2
                    majority_decisions.append(is_majority)
                majority_rate = np.mean(majority_decisions)
            else:
                majority_rate = 1.0  # With 2 estimators, any agreement is majority
                
            voting_analysis['agreement'] = {
                'unanimous_rate': unanimous_rate,
                'majority_rate': majority_rate,
                'n_estimators': prediction_matrix.shape[1]
            }
            
            logger.info(f"Unanimous agreement rate: {unanimous_rate:.4f}")
            logger.info(f"Majority agreement rate: {majority_rate:.4f}")
            
        # Analyze prediction confidence
        max_proba = np.max(ensemble_proba, axis=1)
        confidence_analysis = {
            'mean_confidence': float(np.mean(max_proba)),
            'std_confidence': float(np.std(max_proba)),
            'min_confidence': float(np.min(max_proba)),
            'max_confidence': float(np.max(max_proba)),
            'confidence_distribution': np.histogram(max_proba, bins=10)[0].tolist()
        }
        
        voting_analysis['confidence'] = confidence_analysis
        
        return voting_analysis
        
    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        # Prepare data
        X_processed, y_processed = self.model._prepare_data(X, y, is_training=True)
        
        # Create fresh model for CV
        cv_model = VotingEnsembleWithSoftmax(self.config)
        cv_model.model = cv_model.builder.build_model((X_processed.shape[1],))
        
        # Perform cross-validation for multiple metrics
        cv_results = {}
        
        for scoring in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
            try:
                scores = cross_val_score(
                    cv_model.model, X_processed, y_processed,
                    cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                    scoring=scoring
                )
                
                cv_results[scoring] = {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'scores': scores.tolist()
                }
                
                logger.info(f"CV {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                
            except Exception as e:
                logger.warning(f"Could not calculate CV {scoring}: {e}")
                
        return cv_results
        
    def generate_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_proba: np.ndarray, X: np.ndarray, timestamp: str) -> None:
        """
        Generate Voting Ensemble specific visualizations.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray): Predicted probabilities
            X (np.ndarray): Features for additional analysis
            timestamp (str): Timestamp for file naming
        """
        logger.info("Generating Voting Ensemble specific visualizations...")
        
        # 1. Confusion Matrix
        try:
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=sorted(list(set(y_true))),
                       yticklabels=sorted(list(set(y_true))))
            plt.title('Voting Ensemble Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f"evaluation_plots/confusion_matrix_{timestamp}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Confusion matrix plot saved")
        except Exception as e:
            logger.warning(f"Could not generate confusion matrix: {e}")
            
        # 2. Base Learner Performance Comparison
        try:
            if hasattr(self.model.model, 'estimators_'):
                base_analysis = self.analyze_base_learners(X, y_true)
                individual_performances = base_analysis.get('individual_performances', {})
                
                if individual_performances:
                    estimator_names = list(individual_performances.keys())
                    accuracies = [individual_performances[name]['accuracy'] for name in estimator_names]
                    f1_scores = [individual_performances[name]['f1_weighted'] for name in estimator_names]
                    
                    # Add ensemble performance
                    ensemble_accuracy = accuracy_score(y_true, y_pred)
                    ensemble_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    
                    estimator_names.append('Ensemble')
                    accuracies.append(ensemble_accuracy)
                    f1_scores.append(ensemble_f1)
                    
                    # Create comparison plot
                    x = np.arange(len(estimator_names))
                    width = 0.35
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7)
                    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.7)
                    
                    ax.set_xlabel('Classifiers')
                    ax.set_ylabel('Score')
                    ax.set_title('Base Learner vs Ensemble Performance')
                    ax.set_xticks(x)
                    ax.set_xticklabels(estimator_names, rotation=45, ha='right')
                    ax.legend()
                    
                    # Add value labels on bars
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{height:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(f"evaluation_plots/base_learner_comparison_{timestamp}.png",
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("Base learner comparison plot saved")
        except Exception as e:
            logger.warning(f"Could not generate base learner comparison plot: {e}")
            
        # 3. Prediction Confidence Distribution
        try:
            max_proba = np.max(y_proba, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.hist(max_proba, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Maximum Prediction Probability (Confidence)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Ensemble Prediction Confidence')
            plt.axvline(max_proba.mean(), color='red', linestyle='--', 
                       label=f'Mean: {max_proba.mean():.3f}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"evaluation_plots/prediction_confidence_{timestamp}.png",
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Prediction confidence plot saved")
        except Exception as e:
            logger.warning(f"Could not generate prediction confidence plot: {e}")
            
        # 4. Voting Agreement Analysis
        try:
            voting_analysis = self.analyze_voting_patterns(X, y_true)
            agreement_data = voting_analysis.get('agreement', {})
            
            if agreement_data:
                categories = ['Unanimous', 'Majority']
                values = [agreement_data.get('unanimous_rate', 0), 
                         agreement_data.get('majority_rate', 0)]
                
                plt.figure(figsize=(8, 6))
                bars = plt.bar(categories, values, color=['skyblue', 'lightcoral'], alpha=0.7)
                plt.ylabel('Agreement Rate')
                plt.title('Voting Agreement Analysis')
                plt.ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(f"evaluation_plots/voting_agreement_{timestamp}.png",
                           dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Voting agreement plot saved")
        except Exception as e:
            logger.warning(f"Could not generate voting agreement plot: {e}")
            
        # 5. Diversity Measures Visualization
        try:
            base_analysis = self.analyze_base_learners(X, y_true)
            diversity_measures = base_analysis.get('diversity_measures', {})
            
            if 'average' in diversity_measures:
                avg_measures = diversity_measures['average']
                
                measures = ['Disagreement', 'Double-Fault', 'Q-Statistic']
                values = [
                    avg_measures.get('mean_disagreement', 0),
                    avg_measures.get('mean_double_fault', 0),
                    abs(avg_measures.get('mean_q_statistic', 0))  # Take absolute value for visualization
                ]
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(measures, values, color=['lightgreen', 'salmon', 'lightblue'], alpha=0.7)
                plt.ylabel('Measure Value')
                plt.title('Ensemble Diversity Measures')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(f"evaluation_plots/diversity_measures_{timestamp}.png",
                           dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Diversity measures plot saved")
        except Exception as e:
            logger.warning(f"Could not generate diversity measures plot: {e}")
            
    def generate_detailed_report(self, evaluation_results: Dict[str, Any], 
                               timestamp: str) -> str:
        """
        Generate a detailed Voting Ensemble evaluation report.
        
        Args:
            evaluation_results (Dict[str, Any]): Evaluation results
            timestamp (str): Timestamp for the report
            
        Returns:
            str: Path to the generated report
        """
        logger.info("Generating detailed Voting Ensemble evaluation report...")
        
        report_content = f"""
# Voting Ensemble Model Evaluation Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Model Type: Voting Ensemble Classifier
- Evaluation Timestamp: {timestamp}
- Voting Strategy: {getattr(self.config, 'voting', 'N/A')}

## Dataset Information
- Total Samples: {evaluation_results.get('total_samples', 'N/A')}
- Number of Features: {evaluation_results.get('num_features', 'N/A')}
- Number of Classes: {evaluation_results.get('num_classes', 'N/A')}

## Performance Metrics

### Overall Ensemble Metrics
"""
        
        # Add basic metrics
        basic_metrics = evaluation_results.get('basic_metrics', {})
        for metric, value in basic_metrics.items():
            if isinstance(value, float):
                report_content += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
                
        # Add base learner analysis
        base_learner_analysis = evaluation_results.get('base_learner_analysis', {})
        individual_performances = base_learner_analysis.get('individual_performances', {})
        
        if individual_performances:
            report_content += "\n### Individual Base Learner Performance\n"
            for name, performance in individual_performances.items():
                report_content += f"\n#### {name}\n"
                report_content += f"- Accuracy: {performance.get('accuracy', 0):.4f}\n"
                report_content += f"- F1-Score (Weighted): {performance.get('f1_weighted', 0):.4f}\n"
                
        # Add diversity analysis
        diversity_measures = base_learner_analysis.get('diversity_measures', {})
        if 'average' in diversity_measures:
            avg_diversity = diversity_measures['average']
            report_content += "\n### Ensemble Diversity Analysis\n"
            report_content += f"- Mean Disagreement: {avg_diversity.get('mean_disagreement', 0):.4f}\n"
            report_content += f"- Mean Double-Fault: {avg_diversity.get('mean_double_fault', 0):.4f}\n"
            report_content += f"- Mean Q-Statistic: {avg_diversity.get('mean_q_statistic', 0):.4f}\n"
            
        # Add voting analysis
        voting_analysis = evaluation_results.get('voting_analysis', {})
        if 'agreement' in voting_analysis:
            agreement = voting_analysis['agreement']
            report_content += "\n### Voting Agreement Analysis\n"
            report_content += f"- Unanimous Agreement Rate: {agreement.get('unanimous_rate', 0):.4f}\n"
            report_content += f"- Majority Agreement Rate: {agreement.get('majority_rate', 0):.4f}\n"
            report_content += f"- Number of Base Learners: {agreement.get('n_estimators', 'N/A')}\n"
            
        if 'confidence' in voting_analysis:
            confidence = voting_analysis['confidence']
            report_content += "\n### Prediction Confidence Analysis\n"
            report_content += f"- Mean Confidence: {confidence.get('mean_confidence', 0):.4f}\n"
            report_content += f"- Confidence Std Dev: {confidence.get('std_confidence', 0):.4f}\n"
            report_content += f"- Min Confidence: {confidence.get('min_confidence', 0):.4f}\n"
            report_content += f"- Max Confidence: {confidence.get('max_confidence', 0):.4f}\n"
            
        # Add cross-validation results
        cv_results = evaluation_results.get('cross_validation', {})
        if cv_results:
            report_content += "\n### Cross-Validation Results\n"
            for metric, values in cv_results.items():
                if isinstance(values, dict):
                    mean_val = values.get('mean', 0)
                    std_val = values.get('std', 0)
                    report_content += f"- {metric.replace('_', ' ').title()}: {mean_val:.4f} (+/- {std_val * 2:.4f})\n"
                    
        # Add per-class metrics
        class_report = evaluation_results.get('per_class_metrics', {}).get('classification_report', {})
        if class_report:
            report_content += "\n### Per-Class Performance\n"
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    report_content += f"\n#### Class: {class_name}\n"
                    report_content += f"- Precision: {metrics.get('precision', 0):.4f}\n"
                    report_content += f"- Recall: {metrics.get('recall', 0):.4f}\n"
                    report_content += f"- F1-Score: {metrics.get('f1-score', 0):.4f}\n"
                    report_content += f"- Support: {metrics.get('support', 0)}\n"
                    
        # Add training information if available
        if hasattr(self.model, 'training_history') and self.model.training_history:
            report_content += "\n### Training Information\n"
            history = self.model.training_history
            report_content += f"- Training Time: {history.get('training_time', 'N/A'):.2f} seconds\n"
            
        report_content += "\n### Model Configuration\n"
        if self.config:
            config_dict = self.config.dict() if hasattr(self.config, 'dict') else vars(self.config)
            for key, value in config_dict.items():
                if key not in ['model_save_path']:  # Skip potentially long paths
                    report_content += f"- {key}: {value}\n"
                    
        report_content += f"""
## Voting Ensemble Specific Analysis

### Ensemble Strategy
- **Voting Type**: {getattr(self.config, 'voting', 'N/A')}
- **Base Learners**: Multiple diverse classifiers combined
- **Decision Making**: {'Soft voting (probability averaging)' if getattr(self.config, 'voting', '') == 'soft' else 'Hard voting (majority rule)'}

### Diversity Benefits
- **Error Reduction**: Different base learners make different errors
- **Robustness**: Ensemble is more robust than individual classifiers
- **Generalization**: Better generalization through model averaging

### Interpretation Notes
- **Unanimous Decisions**: All base learners agree on prediction
- **Majority Decisions**: More than half of base learners agree
- **Disagreement Measure**: Fraction of instances where classifiers disagree
- **Double-Fault**: Fraction where multiple classifiers are wrong together

## Files Generated
- Confusion Matrix: evaluation_plots/confusion_matrix_{timestamp}.png
- Base Learner Comparison: evaluation_plots/base_learner_comparison_{timestamp}.png
- Prediction Confidence: evaluation_plots/prediction_confidence_{timestamp}.png
- Voting Agreement: evaluation_plots/voting_agreement_{timestamp}.png
- Diversity Measures: evaluation_plots/diversity_measures_{timestamp}.png

## Notes
- Voting ensembles combine multiple diverse classifiers
- Performance often exceeds individual base learners
- Diversity among base learners is key to ensemble success
- Soft voting generally performs better when probability estimates are reliable

---
Report generated by Voting Ensemble Evaluation Framework v1.0.0
"""
        
        # Save report
        report_path = Path("evaluation_reports") / f"voting_ensemble_evaluation_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        logger.info(f"Detailed report saved to: {report_path}")
        return str(report_path)
        
    def evaluate_on_dataset(self, data_path: str, 
                           generate_plots: bool = True,
                           generate_report: bool = True,
                           perform_cv: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive Voting Ensemble evaluation on a dataset.
        
        Args:
            data_path (str): Path to evaluation dataset
            generate_plots (bool): Whether to generate visualization plots
            generate_report (bool): Whether to generate detailed report
            perform_cv (bool): Whether to perform cross-validation
            
        Returns:
            Dict[str, Any]: Complete evaluation results
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
            
        logger.info("Starting comprehensive Voting Ensemble model evaluation...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load and prepare data
        X, y, feature_names = self.load_and_prepare_data(data_path)
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        basic_metrics = self.calculate_basic_metrics(y, y_pred, y_proba)
        
        # Calculate per-class metrics
        per_class_metrics = {
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'unique_classes': sorted(list(set(y))),
            'class_distribution': {str(cls): int(np.sum(y == cls)) for cls in sorted(list(set(y)))}
        }
        
        # Analyze base learners
        base_learner_analysis = self.analyze_base_learners(X, y)
        
        # Analyze voting patterns
        voting_analysis = self.analyze_voting_patterns(X, y)
        
        # Perform cross-validation if requested
        cv_results = {}
        if perform_cv:
            cv_results = self.perform_cross_validation(X, y)
            
        # Compile results
        evaluation_results = {
            'timestamp': timestamp,
            'total_samples': len(X),
            'num_features': X.shape[1],
            'num_classes': len(np.unique(y)),
            'basic_metrics': basic_metrics,
            'per_class_metrics': per_class_metrics,
            'base_learner_analysis': base_learner_analysis,
            'voting_analysis': voting_analysis,
            'cross_validation': cv_results,
            'feature_names': feature_names
        }
        
        # Generate visualizations
        if generate_plots:
            self.generate_visualizations(y, y_pred, y_proba, X, timestamp)
            
        # Generate detailed report
        if generate_report:
            report_path = self.generate_detailed_report(evaluation_results, timestamp)
            evaluation_results['report_path'] = report_path
            
        # Save results
        results_path = Path("evaluation_outputs") / f"voting_ensemble_evaluation_results_{timestamp}.joblib"
        joblib.dump(evaluation_results, results_path)
        logger.info(f"Evaluation results saved to: {results_path}")
        
        # Store in instance
        self.evaluation_results = evaluation_results
        
        logger.info("Voting Ensemble evaluation completed successfully!")
        return evaluation_results
        
    def print_summary(self) -> None:
        """Print a summary of the Voting Ensemble evaluation results."""
        if not self.evaluation_results:
            logger.warning("No evaluation results available. Run evaluate_on_dataset() first.")
            return
            
        results = self.evaluation_results
        basic_metrics = results.get('basic_metrics', {})
        base_learner_analysis = results.get('base_learner_analysis', {})
        voting_analysis = results.get('voting_analysis', {})
        
        print("\n" + "=" * 70)
        print("VOTING ENSEMBLE MODEL EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Evaluation Timestamp: {results.get('timestamp', 'N/A')}")
        print(f"Total Samples: {results.get('total_samples', 'N/A')}")
        print(f"Number of Features: {results.get('num_features', 'N/A')}")
        print(f"Number of Classes: {results.get('num_classes', 'N/A')}")
        
        # Base learner information
        individual_performances = base_learner_analysis.get('individual_performances', {})
        if individual_performances:
            print(f"Number of Base Learners: {len(individual_performances)}")
            estimator_names = list(individual_performances.keys())
            print(f"Base Learners: {', '.join(estimator_names)}")
            
        print("\n" + "-" * 30)
        print("PERFORMANCE METRICS")
        print("-" * 30)
        
        for metric, value in basic_metrics.items():
            if isinstance(value, float):
                print(f"{metric.replace('_', ' ').title():<25}: {value:.4f}")
                
        # Voting analysis summary
        if 'agreement' in voting_analysis:
            agreement = voting_analysis['agreement']
            print(f"\nUnanimous Agreement Rate: {agreement.get('unanimous_rate', 0):.4f}")
            print(f"Majority Agreement Rate: {agreement.get('majority_rate', 0):.4f}")
            
        if 'confidence' in voting_analysis:
            confidence = voting_analysis['confidence']
            print(f"Mean Prediction Confidence: {confidence.get('mean_confidence', 0):.4f}")
            
        print("\n" + "-" * 30)
        print("FILES GENERATED")
        print("-" * 30)
        
        timestamp = results.get('timestamp', '')
        print(f"• Evaluation Report: evaluation_reports/voting_ensemble_evaluation_report_{timestamp}.md")
        print(f"• Confusion Matrix: evaluation_plots/confusion_matrix_{timestamp}.png")
        print(f"• Base Learner Comparison: evaluation_plots/base_learner_comparison_{timestamp}.png")
        print(f"• Voting Analysis: evaluation_plots/voting_agreement_{timestamp}.png")
        print(f"• Results Data: evaluation_outputs/voting_ensemble_evaluation_results_{timestamp}.joblib")
        
        print("=" * 70)


def main():
    """Main function for Voting Ensemble model evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Voting Ensemble model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to evaluation dataset CSV file"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip detailed report generation"
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation"
    )
    
    args = parser.parse_args()
    
    try:
        # Create evaluator and load model
        evaluator = VotingEnsembleEvaluator(args.model)
        
        # Run evaluation
        results = evaluator.evaluate_on_dataset(
            args.data,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
            perform_cv=not args.no_cv
        )
        
        # Print summary
        evaluator.print_summary()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
