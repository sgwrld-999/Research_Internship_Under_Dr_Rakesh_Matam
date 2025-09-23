"""
Bayesian optimization for hyperparameter tuning.
"""

import optuna
import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple
import logging
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
import warnings

from ..autoencoder import AutoencoderTrainer
from ..ensemble import StackingEnsemble
from ..utils.logger import get_logger


class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Optimization settings
        self.n_trials = config.get('n_trials', 50)
        self.timeout = config.get('timeout', 3600)  # seconds
        self.search_space = config.get('search_space', {})
        
        # Best results
        self.best_params = None
        self.best_score = None
        self.study = None
        
        # Suppress warnings during optimization
        warnings.filterwarnings('ignore')
    
    def _suggest_autoencoder_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest autoencoder hyperparameters."""
        ae_search_space = self.search_space.get('autoencoder', {})
        
        params = {}
        
        if 'bottleneck_dim' in ae_search_space:
            min_val, max_val = ae_search_space['bottleneck_dim']
            params['bottleneck_dim'] = trial.suggest_int('ae_bottleneck_dim', min_val, max_val)
        
        if 'learning_rate' in ae_search_space:
            min_val, max_val = ae_search_space['learning_rate']
            params['learning_rate'] = trial.suggest_float('ae_learning_rate', min_val, max_val, log=True)
        
        if 'dropout_rate' in ae_search_space:
            min_val, max_val = ae_search_space['dropout_rate']
            params['dropout_rate'] = trial.suggest_float('ae_dropout_rate', min_val, max_val)
        
        if 'batch_size' in ae_search_space:
            min_val, max_val = ae_search_space['batch_size']
            # Suggest powers of 2 for batch size
            power = trial.suggest_int('ae_batch_size_power', 
                                    int(np.log2(min_val)), int(np.log2(max_val)))
            params['batch_size'] = 2 ** power
        
        return params
    
    def _suggest_ensemble_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest ensemble hyperparameters."""
        ensemble_search_space = self.search_space.get('ensemble', {})
        
        params = {}
        
        # LightGBM parameters
        if 'lightgbm_n_estimators' in ensemble_search_space:
            min_val, max_val = ensemble_search_space['lightgbm_n_estimators']
            params['lightgbm_n_estimators'] = trial.suggest_int('lgbm_n_estimators', min_val, max_val)
        
        if 'lightgbm_max_depth' in ensemble_search_space:
            min_val, max_val = ensemble_search_space['lightgbm_max_depth']
            params['lightgbm_max_depth'] = trial.suggest_int('lgbm_max_depth', min_val, max_val)
        
        # XGBoost parameters
        if 'xgb_n_estimators' in ensemble_search_space:
            min_val, max_val = ensemble_search_space['xgb_n_estimators']
            params['xgb_n_estimators'] = trial.suggest_int('xgb_n_estimators', min_val, max_val)
        
        if 'xgb_max_depth' in ensemble_search_space:
            min_val, max_val = ensemble_search_space['xgb_max_depth']
            params['xgb_max_depth'] = trial.suggest_int('xgb_max_depth', min_val, max_val)
        
        # Random Forest parameters
        if 'rf_n_estimators' in ensemble_search_space:
            min_val, max_val = ensemble_search_space['rf_n_estimators']
            params['rf_n_estimators'] = trial.suggest_int('rf_n_estimators', min_val, max_val)
        
        if 'rf_max_depth' in ensemble_search_space:
            min_val, max_val = ensemble_search_space['rf_max_depth']
            params['rf_max_depth'] = trial.suggest_int('rf_max_depth', min_val, max_val)
        
        return params
    
    def _update_configs(
        self,
        base_ae_config: Dict[str, Any],
        base_ensemble_config: Dict[str, Any],
        ae_params: Dict[str, Any],
        ensemble_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Update configurations with suggested parameters."""
        
        # Update autoencoder config
        ae_config = base_ae_config.copy()
        
        if 'bottleneck_dim' in ae_params:
            ae_config['architecture']['bottleneck_dim'] = ae_params['bottleneck_dim']
        
        if 'learning_rate' in ae_params:
            ae_config['training']['learning_rate'] = ae_params['learning_rate']
        
        if 'dropout_rate' in ae_params:
            ae_config['architecture']['dropout_rate'] = ae_params['dropout_rate']
        
        if 'batch_size' in ae_params:
            ae_config['training']['batch_size'] = ae_params['batch_size']
        
        # Update ensemble config
        ensemble_config = base_ensemble_config.copy()
        
        # Update LightGBM parameters
        if 'lightgbm_n_estimators' in ensemble_params:
            ensemble_config['base_learners']['lightgbm']['n_estimators'] = ensemble_params['lightgbm_n_estimators']
        
        if 'lightgbm_max_depth' in ensemble_params:
            ensemble_config['base_learners']['lightgbm']['max_depth'] = ensemble_params['lightgbm_max_depth']
        
        # Update XGBoost parameters
        if 'xgb_n_estimators' in ensemble_params:
            ensemble_config['base_learners']['xgboost']['n_estimators'] = ensemble_params['xgb_n_estimators']
        
        if 'xgb_max_depth' in ensemble_params:
            ensemble_config['base_learners']['xgboost']['max_depth'] = ensemble_params['xgb_max_depth']
        
        # Update Random Forest parameters
        if 'rf_n_estimators' in ensemble_params:
            ensemble_config['base_learners']['random_forest']['n_estimators'] = ensemble_params['rf_n_estimators']
        
        if 'rf_max_depth' in ensemble_params:
            ensemble_config['base_learners']['random_forest']['max_depth'] = ensemble_params['rf_max_depth']
        
        return ae_config, ensemble_config
    
    def _objective_function(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: Dict[int, float],
        base_ae_config: Dict[str, Any],
        base_ensemble_config: Dict[str, Any]
    ) -> float:
        """
        Objective function for optimization.
        
        Returns:
            Validation F1 score to maximize
        """
        try:
            # Suggest hyperparameters
            ae_params = self._suggest_autoencoder_params(trial)
            ensemble_params = self._suggest_ensemble_params(trial)
            
            # Update configurations
            ae_config, ensemble_config = self._update_configs(
                base_ae_config, base_ensemble_config, ae_params, ensemble_params
            )
            
            # Reduce epochs for faster optimization
            ae_config['training']['epochs'] = min(ae_config['training'].get('epochs', 100), 20)
            ae_config['training']['early_stopping_patience'] = 5
            
            # Train autoencoder
            ae_trainer = AutoencoderTrainer(ae_config)
            ae_trainer.train(X_train, y_train, X_val, y_val, class_weights)
            
            # Extract embeddings
            train_embeddings = ae_trainer.extract_embeddings(X_train)
            val_embeddings = ae_trainer.extract_embeddings(X_val)
            
            # Train ensemble
            ensemble = StackingEnsemble(ensemble_config)
            ensemble.fit(train_embeddings, y_train)
            
            # Evaluate on validation set
            y_pred = ensemble.predict(val_embeddings)
            f1 = f1_score(y_val, y_pred, average='binary')
            
            # Log trial result
            self.logger.info(f"Trial {trial.number}: F1 = {f1:.4f}")
            
            return f1
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return poor score for failed trials
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: Dict[int, float],
        base_ae_config: Dict[str, Any],
        base_ensemble_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        """
        Perform Bayesian optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            class_weights: Class weights
            base_ae_config: Base autoencoder configuration
            base_ensemble_config: Base ensemble configuration
            
        Returns:
            Optimized autoencoder config, ensemble config, and best score
        """
        self.logger.info("Starting Bayesian optimization")
        self.logger.info(f"Number of trials: {self.n_trials}")
        self.logger.info(f"Timeout: {self.timeout} seconds")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        
        # Define objective with data
        objective = lambda trial: self._objective_function(
            trial, X_train, y_train, X_val, y_val, class_weights,
            base_ae_config, base_ensemble_config
        )
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[self._log_callback]
        )
        
        # Get best parameters
        best_trial = self.study.best_trial
        self.best_score = best_trial.value
        
        # Extract best parameters
        ae_params = self._suggest_autoencoder_params(best_trial)
        ensemble_params = self._suggest_ensemble_params(best_trial)
        
        # Update configurations with best parameters
        best_ae_config, best_ensemble_config = self._update_configs(
            base_ae_config, base_ensemble_config, ae_params, ensemble_params
        )
        
        self.logger.info(f"Optimization completed. Best F1 score: {self.best_score:.4f}")
        self.logger.info(f"Best autoencoder params: {ae_params}")
        self.logger.info(f"Best ensemble params: {ensemble_params}")
        
        return best_ae_config, best_ensemble_config, self.best_score
    
    def _log_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback for logging optimization progress."""
        if trial.number % 10 == 0:
            self.logger.info(f"Trial {trial.number}: Current best F1 = {study.best_value:.4f}")
    
    def get_optimization_history(self) -> Optional[Dict[str, Any]]:
        """Get optimization history."""
        if self.study is None:
            return None
        
        return {
            'trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'trials_dataframe': self.study.trials_dataframe()
        }
    
    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """Plot optimization history."""
        if self.study is None:
            self.logger.warning("No study available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Plot optimization history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Objective value history
            optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=ax1)
            ax1.set_title('Optimization History')
            
            # Parameter importance
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
            ax2.set_title('Parameter Importance')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Optimization plots saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            self.logger.error(f"Error plotting optimization history: {e}")
