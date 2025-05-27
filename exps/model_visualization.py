import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, 
    make_scorer, 
    cross_val_score, 
    root_mean_squared_error,
    r2_score,
    mean_squared_error
)
from sklearn.model_selection import train_test_split

def plot_predictions_vs_actual(y_true, y_pred, model_name, dataset_type="Training"):
    """Plot predictions vs actual values for a model."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - {dataset_type} Set Predictions vs Actual Values')
    
    # Add metrics to plot
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics_text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}'
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for a model if available."""
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 6))
        importances = model.feature_importances_
        fi_df = pd.Series(importances, index=feature_names).sort_values(ascending=True)
        fi_df.plot(kind='barh')
        plt.title(f'{model_name} - Feature Importances')
        plt.tight_layout()
        plt.show()
    elif hasattr(model, 'coef_'):
        plt.figure(figsize=(12, 6))
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef.ravel()
        fi_df = pd.Series(coef, index=feature_names).sort_values(ascending=True)
        fi_df.plot(kind='barh')
        plt.title(f'{model_name} - Feature Coefficients')
        plt.tight_layout()
        plt.show()

def evaluate_model_with_plots(name: str, model, X, y, cv, num_cols=None, cat_cols=None, test_size=0.2, random_state=42):
    """Evaluate model performance and generate visualizations with multiple metrics."""
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Cross-validation scoring on training set
    cv_metrics = {
        'rmse': make_scorer(root_mean_squared_error, greater_is_better=False),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score, greater_is_better=True)
    }
    
    cv_scores = {}
    for metric_name, scorer in cv_metrics.items():
        scores = cross_val_score(model, X_train, y_train, scoring=scorer, cv=cv, n_jobs=-1)
        if metric_name in ['rmse', 'mae']:
            scores = -scores  # Convert back to positive values
        cv_scores[metric_name] = {
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    print(f"\nCross-validation scores for {name}:")
    for metric, scores in cv_scores.items():
        print(f"{metric.upper()}: {scores['mean']:.3f} ± {scores['std']:.3f}")
    
    # Fit model on training set
    model.fit(X_train, y_train)
    
    # Get predictions for both training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate test set metrics
    test_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'r2': r2_score(y_test, y_test_pred)
    }
    
    print(f"\nTest set metrics for {name}:")
    for metric, value in test_metrics.items():
        print(f"{metric.upper()}: {value:.3f}")
    
    # Plot predictions vs actual for both training and test sets
    plot_predictions_vs_actual(y_train, y_train_pred, name, "Training")
    plot_predictions_vs_actual(y_test, y_test_pred, name, "Test")
    
    # Get feature names and plot feature importance
    if hasattr(model, 'named_steps'):
        feature_names = (
            model.named_steps['prep'].named_transformers_['num'].get_feature_names_out(num_cols).tolist()
            + model.named_steps['prep'].named_transformers_['cat']
            .named_steps['ohe']
            .get_feature_names_out(cat_cols)
            .tolist()
        )
        plot_feature_importance(model.named_steps['model'], feature_names, name)
    else:
        feature_names = X.columns.tolist()
        plot_feature_importance(model, feature_names, name)
    
    return {
        "model": name,
        "cv_scores": cv_scores,
        "test_metrics": test_metrics
    } 