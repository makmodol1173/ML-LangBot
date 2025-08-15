"""
Curriculum Management Module
Handles ML curriculum structure and topic organization
"""

from typing import Dict, List


class CurriculumManager:
    """Manages the ML curriculum and topic organization"""
    
    def __init__(self):
        self.curriculum = self._initialize_curriculum()
        self.quick_topics = [
            "Random Forest Classification", 
            "K-Means", 
            "PCA", 
            "SVM", 
            "ANN"
        ]
    
    def _initialize_curriculum(self) -> Dict[str, List[str]]:
        """Initialize the ML curriculum structure"""
        return {
            "Data Preprocessing": [
                "Missing Data", 
                "Encoding", 
                "Feature Scaling"
            ],
            "Regression": [
                "Linear Regression", 
                "Polynomial Regression", 
                "SVR", 
                "Decision Tree Regression", 
                "Random Forest Regression"
            ],
            "Classification": [
                "Logistic Regression", 
                "K-NN", 
                "SVM", 
                "Naive Bayes", 
                "Decision Tree Classification", 
                "Random Forest Classification"
            ],
            "Clustering": [
                "K-Means", 
                "Hierarchical Clustering"
            ],
            "Association Rule Learning": [
                "Apriori", 
                "Eclat"
            ],
            "Reinforcement Learning": [
                "UCB", 
                "Thompson Sampling", 
                "Q-Learning"
            ],
            "NLP": [
                "Bag-of-Words", 
                "TF-IDF", 
                "Word2Vec", 
                "BERT"
            ],
            "Deep Learning": [
                "ANN", 
                "CNN", 
                "RNN", 
                "LSTM"
            ],
            "Dimensionality Reduction": [
                "PCA", 
                "LDA", 
                "Kernel PCA"
            ],
            "Model Selection & Boosting": [
                "Cross Validation", 
                "Grid Search", 
                "XGBoost", 
                "AdaBoost"
            ]
        }
    
    def get_curriculum(self) -> Dict[str, List[str]]:
        """Get the complete curriculum"""
        return self.curriculum
    
    def get_categories(self) -> List[str]:
        """Get all curriculum categories"""
        return list(self.curriculum.keys())
    
    def get_algorithms_by_category(self, category: str) -> List[str]:
        """Get algorithms for a specific category"""
        return self.curriculum.get(category, [])
    
    def get_quick_topics(self) -> List[str]:
        """Get quick access topics"""
        return self.quick_topics
    
    def is_valid_topic(self, topic: str) -> bool:
        """Check if a topic exists in the curriculum"""
        for algorithms in self.curriculum.values():
            if topic in algorithms:
                return True
        return topic in self.quick_topics
    
    def search_topics(self, query: str) -> List[str]:
        """Search for topics matching the query"""
        query_lower = query.lower()
        matching_topics = []
        
        for algorithms in self.curriculum.values():
            for algorithm in algorithms:
                if query_lower in algorithm.lower():
                    matching_topics.append(algorithm)
        
        return matching_topics
