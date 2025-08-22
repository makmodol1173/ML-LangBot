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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
from typing import Dict, Any, Optional
import streamlit as st


class MLVisualizer:
    """Creates interactive visualizations for ML algorithms"""
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # DATA PREPROCESSING VISUALIZATIONS
    def create_missing_data_viz(self) -> go.Figure:
        """Visualize missing data patterns"""
        # Generate sample data with missing values
        np.random.seed(42)
        data = pd.DataFrame({
            'Feature_A': np.random.randn(100),
            'Feature_B': np.random.randn(100),
            'Feature_C': np.random.randn(100),
            'Feature_D': np.random.randn(100)
        })
        
        # Introduce missing values
        missing_indices = np.random.choice(100, 20, replace=False)
        data.loc[missing_indices[:10], 'Feature_A'] = np.nan
        data.loc[missing_indices[10:15], 'Feature_B'] = np.nan
        data.loc[missing_indices[15:], 'Feature_C'] = np.nan
        
        # Calculate missing percentages
        missing_pct = data.isnull().sum() / len(data) * 100
        
        fig = go.Figure(data=[
            go.Bar(x=missing_pct.index, y=missing_pct.values, marker_color=self.colors[0])
        ])
        
        fig.update_layout(
            title="Missing Data Analysis",
            xaxis_title="Features",
            yaxis_title="Missing Percentage (%)",
            template="plotly_white"
        )
        
        return fig
    
    def create_feature_scaling_viz(self) -> go.Figure:
        """Visualize feature scaling effects"""
        # Generate sample data with different scales
        np.random.seed(42)
        feature1 = np.random.normal(100, 15, 100)  # Large scale
        feature2 = np.random.normal(0.5, 0.1, 100)  # Small scale
        
        # Apply scaling
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(np.column_stack([feature1, feature2]))
        
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=feature1, y=feature2,
            mode='markers',
            name='Original Data',
            marker=dict(color=self.colors[0], size=8)
        ))
        
        # Scaled data (offset for visibility)
        fig.add_trace(go.Scatter(
            x=scaled_data[:, 0], y=scaled_data[:, 1],
            mode='markers',
            name='Scaled Data',
            marker=dict(color=self.colors[1], size=8)
        ))
        
        fig.update_layout(
            title="Feature Scaling Comparison",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            template="plotly_white"
        )
        
        return fig
    
    # REGRESSION VISUALIZATIONS
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
    
    def create_polynomial_regression_viz(self) -> go.Figure:
        """Create polynomial regression visualization"""
        np.random.seed(42)
        X = np.linspace(-2, 2, 100).reshape(-1, 1)
        y = 0.5 * X.flatten()**3 + X.flatten()**2 + 0.5 * X.flatten() + np.random.randn(100) * 0.3
        
        # Polynomial features
        poly_features = PolynomialFeatures(degree=3)
        X_poly = poly_features.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=X.flatten(), y=y,
            mode='markers',
            name='Data Points',
            marker=dict(color=self.colors[0], size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=X.flatten(), y=y_pred,
            mode='lines',
            name='Polynomial Fit',
            line=dict(color=self.colors[1], width=3)
        ))
        
        fig.update_layout(
            title="Polynomial Regression Visualization",
            xaxis_title="Feature (X)",
            yaxis_title="Target (y)",
            template="plotly_white"
        )
        
        return fig
    
    def create_decision_tree_regression_viz(self) -> go.Figure:
        """Create decision tree regression visualization"""
        np.random.seed(42)
        X = np.sort(5 * np.random.rand(80, 1), axis=0)
        y = np.sin(X).ravel() + np.random.randn(80) * 0.1
        
        model = DecisionTreeRegressor(max_depth=3)
        model.fit(X, y)
        
        X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
        y_pred = model.predict(X_test)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=X.flatten(), y=y,
            mode='markers',
            name='Training Data',
            marker=dict(color=self.colors[0], size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=X_test.flatten(), y=y_pred,
            mode='lines',
            name='Decision Tree',
            line=dict(color=self.colors[1], width=3)
        ))
        
        fig.update_layout(
            title="Decision Tree Regression",
            xaxis_title="Feature (X)",
            yaxis_title="Target (y)",
            template="plotly_white"
        )
        
        return fig
    
    # CLASSIFICATION VISUALIZATIONS
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
    
    def create_knn_viz(self) -> go.Figure:
        """Create K-NN classification visualization"""
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                 n_informative=2, random_state=42, n_clusters_per_class=1)
        
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X, y)
        
        # Create decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig = go.Figure()
        
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            showscale=False,
            opacity=0.3,
            colorscale='Viridis'
        ))
        
        for class_val in [0, 1]:
            mask = y == class_val
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1],
                mode='markers',
                name=f'Class {class_val}',
                marker=dict(color=self.colors[class_val], size=8)
            ))
        
        fig.update_layout(
            title="K-NN Classification (k=5)",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            template="plotly_white"
        )
        
        return fig
    
    # CLUSTERING VISUALIZATIONS
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
    
    def create_hierarchical_clustering_viz(self) -> go.Figure:
        """Create hierarchical clustering visualization"""
        X, _ = make_blobs(n_samples=50, centers=3, n_features=2, 
                         random_state=42, cluster_std=1.0)
        
        model = AgglomerativeClustering(n_clusters=3)
        labels = model.fit_predict(X)
        
        fig = go.Figure()
        
        for i in range(3):
            mask = labels == i
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1],
                mode='markers',
                name=f'Cluster {i}',
                marker=dict(color=self.colors[i], size=10)
            ))
        
        fig.update_layout(
            title="Hierarchical Clustering Visualization",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            template="plotly_white"
        )
        
        return fig
    
    # DIMENSIONALITY REDUCTION VISUALIZATIONS
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
    
    def create_lda_viz(self) -> go.Figure:
        """Create LDA visualization"""
        X, y = make_classification(n_samples=200, n_features=4, n_redundant=0, 
                                 n_informative=2, random_state=42, n_classes=3)
        
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_lda = lda.fit_transform(X, y)
        
        fig = go.Figure()
        
        for class_val in [0, 1, 2]:
            mask = y == class_val
            fig.add_trace(go.Scatter(
                x=X_lda[mask, 0], y=X_lda[mask, 1],
                mode='markers',
                name=f'Class {class_val}',
                marker=dict(color=self.colors[class_val], size=8)
            ))
        
        fig.update_layout(
            title="Linear Discriminant Analysis (LDA)",
            xaxis_title="LD1",
            yaxis_title="LD2",
            template="plotly_white"
        )
        
        return fig
    
    # MODEL SELECTION & BOOSTING VISUALIZATIONS
    def create_cross_validation_viz(self) -> go.Figure:
        """Create cross-validation visualization"""
        X, y = make_classification(n_samples=100, n_features=2, random_state=42)
        
        models = ['Logistic Reg', 'Random Forest', 'SVM', 'K-NN', 'Naive Bayes']
        cv_scores = []
        
        for model_name in models:
            if model_name == 'Logistic Reg':
                model = LogisticRegression()
            elif model_name == 'Random Forest':
                model = RandomForestClassifier()
            elif model_name == 'SVM':
                model = SVC()
            elif model_name == 'K-NN':
                model = KNeighborsClassifier()
            else:
                model = GaussianNB()
            
            scores = cross_val_score(model, X, y, cv=5)
            cv_scores.append(scores.mean())
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=cv_scores, marker_color=self.colors[:len(models)])
        ])
        
        fig.update_layout(
            title="Cross-Validation Scores Comparison",
            xaxis_title="Models",
            yaxis_title="CV Score",
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
    
    # NLP VISUALIZATIONS
    def create_bow_viz(self) -> go.Figure:
        """Create Bag of Words visualization"""
        # Sample text data
        documents = [
            "machine learning is great",
            "deep learning neural networks",
            "natural language processing",
            "machine learning algorithms",
            "neural networks deep learning"
        ]
        
        # Simple word frequency
        all_words = ' '.join(documents).split()
        word_freq = pd.Series(all_words).value_counts().head(8)
        
        fig = go.Figure(data=[
            go.Bar(x=word_freq.index, y=word_freq.values, marker_color=self.colors[0])
        ])
        
        fig.update_layout(
            title="Bag of Words - Word Frequency",
            xaxis_title="Words",
            yaxis_title="Frequency",
            template="plotly_white"
        )
        
        return fig
    
    # PLACEHOLDER METHODS FOR COMPLEX ALGORITHMS
    def create_encoding_viz(self) -> Optional[go.Figure]:
        """Encoding visualization - placeholder"""
        return None
    
    def create_svr_viz(self) -> Optional[go.Figure]:
        """SVR visualization - placeholder"""
        return None
    
    def create_svm_viz(self) -> Optional[go.Figure]:
        """SVM visualization - placeholder"""
        return None
    
    def create_naive_bayes_viz(self) -> Optional[go.Figure]:
        """Naive Bayes visualization - placeholder"""
        return None
    
    def create_decision_tree_classification_viz(self) -> Optional[go.Figure]:
        """Decision Tree Classification visualization - placeholder"""
        return None
    
    def create_random_forest_classification_viz(self) -> Optional[go.Figure]:
        """Random Forest Classification visualization - placeholder"""
        return None
    
    def create_apriori_viz(self) -> Optional[go.Figure]:
        """Apriori visualization - placeholder"""
        return None
    
    def create_eclat_viz(self) -> Optional[go.Figure]:
        """Eclat visualization - placeholder"""
        return None
    
    def create_ucb_viz(self) -> Optional[go.Figure]:
        """UCB visualization - placeholder"""
        return None
    
    def create_thompson_sampling_viz(self) -> Optional[go.Figure]:
        """Thompson Sampling visualization - placeholder"""
        return None
    
    def create_q_learning_viz(self) -> Optional[go.Figure]:
        """Q-Learning visualization - placeholder"""
        return None
    
    def create_tfidf_viz(self) -> Optional[go.Figure]:
        """TF-IDF visualization - placeholder"""
        return None
    
    def create_word2vec_viz(self) -> Optional[go.Figure]:
        """Word2Vec visualization - placeholder"""
        return None
    
    def create_bert_viz(self) -> Optional[go.Figure]:
        """BERT visualization - placeholder"""
        return None
    
    def create_ann_viz(self) -> Optional[go.Figure]:
        """ANN visualization - placeholder"""
        return None
    
    def create_cnn_viz(self) -> Optional[go.Figure]:
        """CNN visualization - placeholder"""
        return None
    
    def create_rnn_viz(self) -> Optional[go.Figure]:
        """RNN visualization - placeholder"""
        return None
    
    def create_lstm_viz(self) -> Optional[go.Figure]:
        """LSTM visualization - placeholder"""
        return None
    
    def create_kernel_pca_viz(self) -> Optional[go.Figure]:
        """Kernel PCA visualization - placeholder"""
        return None
    
    def create_grid_search_viz(self) -> Optional[go.Figure]:
        """Grid Search visualization - placeholder"""
        return None
    
    def create_xgboost_viz(self) -> Optional[go.Figure]:
        """XGBoost visualization - placeholder"""
        return None
    
    def create_adaboost_viz(self) -> Optional[go.Figure]:
        """AdaBoost visualization - placeholder"""
        return None
    
    def get_algorithm_visualization(self, algorithm: str) -> Optional[go.Figure]:
        """Get visualization for a specific algorithm"""
        algorithm_lower = algorithm.lower().replace(' ', '_').replace('-', '_')
        
        algorithm_map = {
            # Data Preprocessing - with actual visualizations
            'missing_data': self.create_missing_data_viz,
            'feature_scaling': self.create_feature_scaling_viz,
            
            # Regression - with actual visualizations
            'linear_regression': self.create_linear_regression_viz,
            'polynomial_regression': self.create_polynomial_regression_viz,
            'decision_tree_regression': self.create_decision_tree_regression_viz,
            'random_forest_regression': self.create_random_forest_viz,
            
            # Classification - with actual visualizations
            'logistic_regression': self.create_classification_viz,
            'k_nn': self.create_knn_viz,
            'knn': self.create_knn_viz,
            
            # Clustering - with actual visualizations
            'k_means': self.create_clustering_viz,
            'kmeans': self.create_clustering_viz,
            'hierarchical_clustering': self.create_hierarchical_clustering_viz,
            
            # NLP - with actual visualizations
            'bag_of_words': self.create_bow_viz,
            'bow': self.create_bow_viz,
            
            # Dimensionality Reduction - with actual visualizations
            'pca': self.create_pca_viz,
            'lda': self.create_lda_viz,
            
            # Model Selection & Boosting - with actual visualizations
            'cross_validation': self.create_cross_validation_viz,
            'random_forest': self.create_random_forest_viz
        }
        
        # Try exact match first
        if algorithm_lower in algorithm_map:
            return algorithm_map[algorithm_lower]()
        
        # Try partial matches
        for key, method in algorithm_map.items():
            if key in algorithm_lower or algorithm_lower in key:
                return method()
        
        return None
    
    def has_visualization(self, algorithm: str) -> bool:
        """Check if an algorithm has a specific visualization available"""
        algorithm_lower = algorithm.lower().replace(' ', '_').replace('-', '_')
        
        algorithms_with_viz = {
            'missing_data', 'feature_scaling', 'linear_regression', 'polynomial_regression',
            'decision_tree_regression', 'random_forest_regression', 'logistic_regression',
            'k_nn', 'knn', 'k_means', 'kmeans', 'hierarchical_clustering', 'bag_of_words',
            'bow', 'pca', 'lda', 'cross_validation', 'random_forest'
        }
        
        # Check exact match
        if algorithm_lower in algorithms_with_viz:
            return True
        
        # Check partial matches
        for algo in algorithms_with_viz:
            if algo in algorithm_lower or algorithm_lower in algo:
                return True
        
        return False
    
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
