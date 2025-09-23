#!/usr/bin/env python3
"""
Comprehensive model loading and plotting script for the Autoencoder-Stacked-Ensemble pipeline.
This script loads trained models and generates various evaluation curves and visualizations.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score,
    calibration_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelLoader:
    """Load and analyze trained models."""
    
    def __init__(self, models_dir='models', results_dir='results', config_path='config/config.yaml'):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.config_path = Path(config_path)
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.load_config()
        
        # Initialize components
        self.pipeline = None
        self.data_processor = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_proba = None
        
    def load_config(self):
        """Load configuration file."""
        import yaml
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"✓ Configuration loaded from {self.config_path}")
        except Exception as e:
            print(f"✗ Error loading config: {e}")
            self.config = {}
    
    def load_pipeline(self):
        """Load the complete trained pipeline."""
        from src.pipeline import AutoencoderStackedEnsemblePipeline
        
        print("Loading complete pipeline...")
        self.pipeline = AutoencoderStackedEnsemblePipeline(self.config)
        
        # Load and preprocess data
        self.pipeline.load_and_preprocess_data()
        self.X_test = self.pipeline.X_test
        self.y_test = self.pipeline.y_test
        
        # Load trained autoencoder
        if (self.models_dir / 'autoencoder.pth').exists():
            from src.autoencoder.trainer import AutoencoderTrainer
            from src.autoencoder.model import CostSensitiveAutoencoder
            
            ae_trainer = AutoencoderTrainer(self.config['autoencoder'])
            ae_trainer.model = CostSensitiveAutoencoder(
                input_dim=self.X_test.shape[1],
                hidden_dims=self.config['autoencoder']['architecture']['hidden_dims'],
                bottleneck_dim=self.config['autoencoder']['architecture']['bottleneck_dim'],
                dropout_rate=self.config['autoencoder']['architecture']['dropout_rate']
            )
            ae_trainer.load_model(str(self.models_dir / 'autoencoder.pth'))
            self.pipeline.ae_trainer = ae_trainer
            print("✓ Autoencoder loaded")
        
        # Load trained ensemble
        if (self.models_dir / 'ensemble.pkl').exists():
            with open(self.models_dir / 'ensemble.pkl', 'rb') as f:
                ensemble = pickle.load(f)
            self.pipeline.ensemble = ensemble
            print("✓ Ensemble loaded")
        
        # Extract embeddings and make predictions
        if self.pipeline.ae_trainer and self.pipeline.ensemble:
            test_embeddings = self.pipeline.ae_trainer.extract_embeddings(self.X_test)
            self.y_pred = self.pipeline.ensemble.predict(test_embeddings)
            self.y_proba = self.pipeline.ensemble.predict_proba(test_embeddings)[:, 1]
            print("✓ Predictions generated")
        
        return self.pipeline
    
    def load_evaluation_results(self):
        """Load saved evaluation results."""
        results_file = self.results_dir / 'evaluation_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return None
    
    def plot_confusion_matrix(self, save_plot=True):
        """Plot enhanced confusion matrix."""
        if self.y_test is None or self.y_pred is None:
            print("✗ No predictions available for confusion matrix")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Normal', 'Attack'], 
                   yticklabels=['Normal', 'Attack'])
        ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        
        # Normalized confusion matrix
        cm_norm = confusion_matrix(self.y_test, self.y_pred, normalize='true')
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=ax2,
                   xticklabels=['Normal', 'Attack'], 
                   yticklabels=['Normal', 'Attack'])
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        if save_plot:
            plt.savefig(self.results_dir / 'enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
            print(f"✓ Enhanced confusion matrix saved to {self.results_dir / 'enhanced_confusion_matrix.png'}")
        plt.show()
    
    def plot_roc_curves(self, save_plot=True):
        """Plot ROC curves for ensemble and individual models."""
        if self.y_test is None or self.pipeline is None:
            print("✗ No data available for ROC curves")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Main ensemble ROC
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=3, label=f'Stacking Ensemble (AUC = {roc_auc:.3f})', color='red')
        
        # Individual base learners ROC
        if hasattr(self.pipeline, 'ensemble') and self.pipeline.ensemble:
            test_embeddings = self.pipeline.ae_trainer.extract_embeddings(self.X_test)
            
            colors = ['blue', 'green', 'orange', 'purple']
            base_learners = ['random_forest', 'lightgbm', 'xgboost', 'catboost']
            
            for i, (learner_name, color) in enumerate(zip(base_learners, colors)):
                if hasattr(self.pipeline.ensemble, 'base_learners') and learner_name in self.pipeline.ensemble.base_learners:
                    learner = self.pipeline.ensemble.base_learners[learner_name]
                    try:
                        if hasattr(learner, 'predict_proba'):
                            y_proba_base = learner.predict_proba(test_embeddings)[:, 1]
                        else:
                            y_proba_base = learner.decision_function(test_embeddings)
                        
                        fpr_base, tpr_base, _ = roc_curve(self.y_test, y_proba_base)
                        roc_auc_base = auc(fpr_base, tpr_base)
                        ax.plot(fpr_base, tpr_base, linewidth=2, linestyle='--', 
                               label=f'{learner_name.upper()} (AUC = {roc_auc_base:.3f})', color=color)
                    except Exception as e:
                        print(f"Could not plot ROC for {learner_name}: {e}")
        
        # Random classifier line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plot:
            plt.savefig(self.results_dir / 'comprehensive_roc_curves.png', dpi=300, bbox_inches='tight')
            print(f"✓ ROC curves saved to {self.results_dir / 'comprehensive_roc_curves.png'}")
        plt.show()
    
    def plot_precision_recall_curves(self, save_plot=True):
        """Plot Precision-Recall curves."""
        if self.y_test is None or self.y_proba is None:
            print("✗ No data available for PR curves")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Main ensemble PR curve
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba)
        avg_precision = average_precision_score(self.y_test, self.y_proba)
        ax.plot(recall, precision, linewidth=3, label=f'Stacking Ensemble (AP = {avg_precision:.3f})', color='red')
        
        # Individual base learners PR curves
        if hasattr(self.pipeline, 'ensemble') and self.pipeline.ensemble:
            test_embeddings = self.pipeline.ae_trainer.extract_embeddings(self.X_test)
            
            colors = ['blue', 'green', 'orange', 'purple']
            base_learners = ['random_forest', 'lightgbm', 'xgboost', 'catboost']
            
            for i, (learner_name, color) in enumerate(zip(base_learners, colors)):
                if hasattr(self.pipeline.ensemble, 'base_learners') and learner_name in self.pipeline.ensemble.base_learners:
                    learner = self.pipeline.ensemble.base_learners[learner_name]
                    try:
                        if hasattr(learner, 'predict_proba'):
                            y_proba_base = learner.predict_proba(test_embeddings)[:, 1]
                        else:
                            y_proba_base = learner.decision_function(test_embeddings)
                        
                        precision_base, recall_base, _ = precision_recall_curve(self.y_test, y_proba_base)
                        avg_precision_base = average_precision_score(self.y_test, y_proba_base)
                        ax.plot(recall_base, precision_base, linewidth=2, linestyle='--', 
                               label=f'{learner_name.upper()} (AP = {avg_precision_base:.3f})', color=color)
                    except Exception as e:
                        print(f"Could not plot PR for {learner_name}: {e}")
        
        # Baseline (random classifier)
        baseline = np.sum(self.y_test) / len(self.y_test)
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1, alpha=0.5, 
                  label=f'Random Classifier (AP = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plot:
            plt.savefig(self.results_dir / 'comprehensive_pr_curves.png', dpi=300, bbox_inches='tight')
            print(f"✓ PR curves saved to {self.results_dir / 'comprehensive_pr_curves.png'}")
        plt.show()
    
    def plot_calibration_curve(self, save_plot=True):
        """Plot calibration curve to assess prediction reliability."""
        if self.y_test is None or self.y_proba is None:
            print("✗ No data available for calibration curve")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_test, self.y_proba, n_bins=10, normalize=False
        )
        
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", 
                linewidth=2, label="Autoencoder-Stacked-Ensemble", color='red')
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=1)
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title('Calibration Curve', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Prediction histogram
        ax2.hist(self.y_proba[self.y_test == 0], bins=20, alpha=0.7, 
                label='Normal', color='blue', density=True)
        ax2.hist(self.y_proba[self.y_test == 1], bins=20, alpha=0.7, 
                label='Attack', color='red', density=True)
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Calculate Brier score
        brier_score = brier_score_loss(self.y_test, self.y_proba)
        fig.suptitle(f'Model Calibration Analysis (Brier Score: {brier_score:.4f})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        if save_plot:
            plt.savefig(self.results_dir / 'calibration_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✓ Calibration analysis saved to {self.results_dir / 'calibration_analysis.png'}")
        plt.show()
    
    def plot_learning_curves(self, save_plot=True):
        """Plot autoencoder training curves if available."""
        try:
            # Try to load training history from the autoencoder trainer
            if hasattr(self.pipeline, 'ae_trainer') and self.pipeline.ae_trainer:
                trainer = self.pipeline.ae_trainer
                if hasattr(trainer, 'train_losses') and hasattr(trainer, 'val_losses'):
                    train_losses = trainer.train_losses
                    val_losses = trainer.val_losses
                    
                    if len(train_losses) > 0 and len(val_losses) > 0:
                        epochs = range(1, len(train_losses) + 1)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o')
                        ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s')
                        ax.set_xlabel('Epoch', fontsize=12)
                        ax.set_ylabel('Loss', fontsize=12)
                        ax.set_title('Autoencoder Training Curves', fontsize=14, fontweight='bold')
                        ax.legend(fontsize=10)
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        if save_plot:
                            plt.savefig(self.results_dir / 'autoencoder_learning_curves.png', dpi=300, bbox_inches='tight')
                            print(f"✓ Learning curves saved to {self.results_dir / 'autoencoder_learning_curves.png'}")
                        plt.show()
                        return
            
            print("⚠ No training history available for learning curves")
            
        except Exception as e:
            print(f"✗ Error plotting learning curves: {e}")
    
    def plot_feature_importance(self, save_plot=True):
        """Plot feature importance for base learners and meta-learner."""
        if not hasattr(self.pipeline, 'ensemble') or not self.pipeline.ensemble:
            print("✗ No ensemble available for feature importance")
            return
        
        try:
            ensemble = self.pipeline.ensemble
            
            # Meta-learner coefficients (feature importance)
            if hasattr(ensemble, 'meta_learner') and hasattr(ensemble.meta_learner, 'coef_'):
                meta_coef = np.abs(ensemble.meta_learner.coef_[0])
                base_learner_names = ['Random Forest', 'LightGBM', 'XGBoost', 'CatBoost']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(base_learner_names, meta_coef, color=['blue', 'green', 'orange', 'purple'])
                ax.set_ylabel('Absolute Coefficient', fontsize=12)
                ax.set_title('Meta-Learner Feature Importance\n(Base Learner Contributions)', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, coef in zip(bars, meta_coef):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{coef:.3f}', ha='center', va='bottom', fontweight='bold')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                if save_plot:
                    plt.savefig(self.results_dir / 'meta_learner_importance.png', dpi=300, bbox_inches='tight')
                    print(f"✓ Meta-learner importance saved to {self.results_dir / 'meta_learner_importance.png'}")
                plt.show()
            else:
                print("⚠ Meta-learner coefficients not available")
                
        except Exception as e:
            print(f"✗ Error plotting feature importance: {e}")
    
    def plot_threshold_analysis(self, save_plot=True):
        """Plot threshold analysis for optimal decision boundary."""
        if self.y_test is None or self.y_proba is None:
            print("✗ No data available for threshold analysis")
            return
        
        # Calculate metrics for different thresholds
        thresholds = np.linspace(0, 1, 101)
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_proba >= threshold).astype(int)
            
            # Calculate metrics
            tp = np.sum((self.y_test == 1) & (y_pred_thresh == 1))
            fp = np.sum((self.y_test == 0) & (y_pred_thresh == 1))
            tn = np.sum((self.y_test == 0) & (y_pred_thresh == 0))
            fn = np.sum((self.y_test == 1) & (y_pred_thresh == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            accuracies.append(accuracy)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(thresholds, precisions, label='Precision', linewidth=2, color='blue')
        ax.plot(thresholds, recalls, label='Recall', linewidth=2, color='red')
        ax.plot(thresholds, f1_scores, label='F1-Score', linewidth=2, color='green')
        ax.plot(thresholds, accuracies, label='Accuracy', linewidth=2, color='orange')
        
        # Find optimal threshold (max F1)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        ax.axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=2, 
                  label=f'Optimal Threshold = {optimal_threshold:.3f}')
        
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Threshold Analysis for Optimal Decision Boundary', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        if save_plot:
            plt.savefig(self.results_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✓ Threshold analysis saved to {self.results_dir / 'threshold_analysis.png'}")
        plt.show()
        
        print(f"📊 Optimal threshold: {optimal_threshold:.3f}")
        print(f"📊 At optimal threshold:")
        print(f"   - Precision: {precisions[optimal_idx]:.3f}")
        print(f"   - Recall: {recalls[optimal_idx]:.3f}")
        print(f"   - F1-Score: {f1_scores[optimal_idx]:.3f}")
        print(f"   - Accuracy: {accuracies[optimal_idx]:.3f}")
    
    def plot_embedding_analysis(self, save_plot=True):
        """Plot autoencoder embedding analysis."""
        if not hasattr(self.pipeline, 'ae_trainer') or not self.pipeline.ae_trainer:
            print("✗ No autoencoder available for embedding analysis")
            return
        
        try:
            # Extract embeddings for a sample of test data
            sample_size = min(1000, len(self.X_test))
            sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test[sample_indices]
            y_sample = self.y_test[sample_indices]
            
            embeddings = self.pipeline.ae_trainer.extract_embeddings(X_sample)
            
            # PCA for visualization (2D)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 2D embedding visualization
            scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                 c=y_sample, cmap='viridis', alpha=0.6, s=20)
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            ax1.set_title('Autoencoder Embeddings (PCA Projection)', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax1, label='Class (0=Normal, 1=Attack)')
            ax1.grid(True, alpha=0.3)
            
            # Embedding distribution by dimension
            embedding_dims = embeddings.shape[1]
            normal_embeddings = embeddings[y_sample == 0]
            attack_embeddings = embeddings[y_sample == 1]
            
            dim_to_plot = min(4, embedding_dims)  # Plot first 4 dimensions
            positions = np.arange(dim_to_plot)
            width = 0.35
            
            normal_means = np.mean(normal_embeddings[:, :dim_to_plot], axis=0)
            attack_means = np.mean(attack_embeddings[:, :dim_to_plot], axis=0)
            normal_stds = np.std(normal_embeddings[:, :dim_to_plot], axis=0)
            attack_stds = np.std(attack_embeddings[:, :dim_to_plot], axis=0)
            
            bars1 = ax2.bar(positions - width/2, normal_means, width, 
                           yerr=normal_stds, label='Normal', color='blue', alpha=0.7)
            bars2 = ax2.bar(positions + width/2, attack_means, width,
                           yerr=attack_stds, label='Attack', color='red', alpha=0.7)
            
            ax2.set_xlabel('Embedding Dimension', fontsize=12)
            ax2.set_ylabel('Mean Value ± Std', fontsize=12)
            ax2.set_title('Embedding Statistics by Class', fontsize=14, fontweight='bold')
            ax2.set_xticks(positions)
            ax2.set_xticklabels([f'Dim {i+1}' for i in range(dim_to_plot)])
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            if save_plot:
                plt.savefig(self.results_dir / 'embedding_analysis.png', dpi=300, bbox_inches='tight')
                print(f"✓ Embedding analysis saved to {self.results_dir / 'embedding_analysis.png'}")
            plt.show()
            
        except Exception as e:
            print(f"✗ Error plotting embedding analysis: {e}")
    
    def generate_all_plots(self):
        """Generate all available plots."""
        print("\n🎨 GENERATING COMPREHENSIVE EVALUATION PLOTS")
        print("=" * 60)
        
        # Load pipeline and models
        try:
            self.load_pipeline()
        except Exception as e:
            print(f"✗ Error loading pipeline: {e}")
            return
        
        # Generate all plots
        plot_methods = [
            ("Confusion Matrix", self.plot_confusion_matrix),
            ("ROC Curves", self.plot_roc_curves),
            ("Precision-Recall Curves", self.plot_precision_recall_curves),
            ("Calibration Analysis", self.plot_calibration_curve),
            ("Threshold Analysis", self.plot_threshold_analysis),
            ("Feature Importance", self.plot_feature_importance),
            ("Embedding Analysis", self.plot_embedding_analysis),
            ("Learning Curves", self.plot_learning_curves),
        ]
        
        successful_plots = 0
        for plot_name, plot_method in plot_methods:
            try:
                print(f"\n📊 Generating {plot_name}...")
                plot_method()
                successful_plots += 1
            except Exception as e:
                print(f"✗ Error generating {plot_name}: {e}")
        
        print(f"\n✅ Successfully generated {successful_plots}/{len(plot_methods)} plots")
        print(f"📁 All plots saved to: {self.results_dir}")
    
    def print_model_summary(self):
        """Print comprehensive model summary."""
        print("\n🔍 MODEL ANALYSIS SUMMARY")
        print("=" * 50)
        
        # Load evaluation results
        results = self.load_evaluation_results()
        if results:
            print("\n📊 PERFORMANCE METRICS:")
            for model_name, metrics in results.items():
                if isinstance(metrics, dict):
                    print(f"\n{model_name}:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"  {metric}: {value:.4f}")
        
        # Model architecture info
        if hasattr(self.pipeline, 'ae_trainer') and self.pipeline.ae_trainer:
            print(f"\n🧠 AUTOENCODER ARCHITECTURE:")
            print(f"  Input Dimension: {self.X_test.shape[1] if self.X_test is not None else 'N/A'}")
            print(f"  Bottleneck Dimension: {self.config.get('autoencoder', {}).get('architecture', {}).get('bottleneck_dim', 'N/A')}")
            print(f"  Hidden Layers: {self.config.get('autoencoder', {}).get('architecture', {}).get('hidden_dims', 'N/A')}")
        
        if hasattr(self.pipeline, 'ensemble') and self.pipeline.ensemble:
            print(f"\n🎯 ENSEMBLE COMPOSITION:")
            if hasattr(self.pipeline.ensemble, 'base_learners'):
                for name in self.pipeline.ensemble.base_learners.keys():
                    print(f"  ✓ {name.upper()}")


def main():
    """Main execution function."""
    print("🚀 AUTOENCODER-STACKED-ENSEMBLE MODEL ANALYZER")
    print("=" * 60)
    
    # Initialize model loader
    loader = ModelLoader()
    
    # Print model summary
    loader.print_model_summary()
    
    # Generate all plots
    loader.generate_all_plots()
    
    print("\n🎉 ANALYSIS COMPLETED!")
    print("=" * 60)
    print(f"📁 Check the '{loader.results_dir}' directory for all generated plots.")


if __name__ == "__main__":
    main()
