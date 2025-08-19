"""
Visualization Module
Handles charts, graphs, and algorithm visualizations using Plotly and Matplotlib
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from typing import Dict, Any, Optional
import streamlit as st


class MLVisualizer:
    """Creates interactive visualizations for ML algorithms"""
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def create_linear_regression_viz(self) -> go.Figure:
        """Create interactive linear regression visualization"""
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(100, 1) * 2
        y = 3 * X.flatten() + 2 + np.random.randn(100) * 0.5
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Create plot
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=X.flatten(), y=y,
            mode='markers',
            name='Data Points',
            marker=dict(color=self.colors[0], size=8)
        ))
        
        # Add regression line
        fig.add_trace(go.Scatter(
            x=X.flatten(), y=y_pred,
            mode='lines',
            name='Regression Line',
            line=dict(color=self.colors[1], width=3)
        ))
        
        fig.update_layout(
            title="Linear Regression Visualization",
            xaxis_title="Feature (X)",
            yaxis_title="Target (y)",
            template="plotly_white"
        )
        
        return fig
    
    def create_classification_viz(self) -> go.Figure:
        """Create classification boundary visualization"""
        # Generate sample data
        X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                 n_informative=2, random_state=42, n_clusters_per_class=1)
        
        # Fit model
        model = LogisticRegression()
        model.fit(X, y)
        
        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        # Create plot
        fig = go.Figure()
        
        # Add decision boundary
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            showscale=False,
            opacity=0.3,
            colorscale='RdYlBu'
        ))
        
        # Add data points
        for class_val in [0, 1]:
            mask = y == class_val
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1],
                mode='markers',
                name=f'Class {class_val}',
                marker=dict(color=self.colors[class_val], size=8)
            ))
        
        fig.update_layout(
            title="Classification Decision Boundary",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            template="plotly_white"
        )
        
        return fig
    
    def create_clustering_viz(self) -> go.Figure:
        """Create K-Means clustering visualization"""
        # Generate sample data
        X, _ = make_blobs(n_samples=300, centers=4, n_features=2, 
                         random_state=42, cluster_std=0.60)
        
        # Fit K-Means
        kmeans = KMeans(n_clusters=4, random_state=42)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_
        
        # Create plot
        fig = go.Figure()
        
        # Add data points
        for i in range(4):
            mask = labels == i
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1],
                mode='markers',
                name=f'Cluster {i}',
                marker=dict(color=self.colors[i], size=8)
            ))
        
        # Add centroids
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1],
            mode='markers',
            name='Centroids',
            marker=dict(color='black', size=15, symbol='x')
        ))
        
        fig.update_layout(
            title="K-Means Clustering Visualization",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            template="plotly_white"
        )
        
        return fig
    
    def create_pca_viz(self) -> go.Figure:
        """Create PCA dimensionality reduction visualization"""
        # Generate sample data
        X, y = make_classification(n_samples=200, n_features=4, n_redundant=0, 
                                 n_informative=2, random_state=42)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create plot
        fig = go.Figure()
        
        # Add data points
        for class_val in [0, 1]:
            mask = y == class_val
            fig.add_trace(go.Scatter(
                x=X_pca[mask, 0], y=X_pca[mask, 1],
                mode='markers',
                name=f'Class {class_val}',
                marker=dict(color=self.colors[class_val], size=8)
            ))
        
        fig.update_layout(
            title=f"PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            template="plotly_white"
        )
        
        return fig
    
    def create_random_forest_viz(self) -> go.Figure:
        """Create Random Forest feature importance visualization"""
        # Generate sample data
        X, y = make_classification(n_samples=200, n_features=8, n_informative=4, 
                                 random_state=42)
        
        # Fit Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance = rf.feature_importances_
        features = [f'Feature {i+1}' for i in range(len(importance))]
        
        # Create plot
        fig = go.Figure(data=[
            go.Bar(x=features, y=importance, marker_color=self.colors[0])
        ])
        
        fig.update_layout(
            title="Random Forest Feature Importance",
            xaxis_title="Features",
            yaxis_title="Importance",
            template="plotly_white"
        )
        
        return fig
    
    def get_algorithm_visualization(self, algorithm: str) -> Optional[go.Figure]:
        """Get visualization for a specific algorithm"""
        algorithm_lower = algorithm.lower()
        
        if 'linear regression' in algorithm_lower:
            return self.create_linear_regression_viz()
        elif any(term in algorithm_lower for term in ['classification', 'logistic', 'svm', 'naive bayes']):
            return self.create_classification_viz()
        elif 'k-means' in algorithm_lower or 'clustering' in algorithm_lower:
            return self.create_clustering_viz()
        elif 'pca' in algorithm_lower:
            return self.create_pca_viz()
        elif 'random forest' in algorithm_lower:
            return self.create_random_forest_viz()
        else:
            return None
    
    def create_algorithm_comparison_chart(self) -> go.Figure:
        """Create algorithm comparison chart"""
        algorithms = ['Linear Reg', 'Logistic Reg', 'Random Forest', 'SVM', 'K-Means']
        accuracy = [0.85, 0.88, 0.92, 0.89, 0.75]
        speed = [0.95, 0.90, 0.70, 0.60, 0.85]
        interpretability = [0.95, 0.85, 0.60, 0.40, 0.80]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=algorithms, y=accuracy,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color=self.colors[0])
        ))
        
        fig.add_trace(go.Scatter(
            x=algorithms, y=speed,
            mode='lines+markers',
            name='Speed',
            line=dict(color=self.colors[1])
        ))
        
        fig.add_trace(go.Scatter(
            x=algorithms, y=interpretability,
            mode='lines+markers',
            name='Interpretability',
            line=dict(color=self.colors[2])
        ))
        
        fig.update_layout(
            title="ML Algorithm Comparison",
            xaxis_title="Algorithms",
            yaxis_title="Score (0-1)",
            template="plotly_white",
            yaxis=dict(range=[0, 1])
        )
        
        return fig
