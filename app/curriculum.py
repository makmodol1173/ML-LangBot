"""
Curriculum Management Module
Handles ML curriculum structure and topic organization
"""

import streamlit as st
from typing import Dict, List, Tuple, Optional
from datetime import datetime


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
        self.topic_difficulty = self._initialize_topic_difficulty()
        self.learning_paths = self._initialize_learning_paths()
        self.prerequisites = self._initialize_prerequisites()
        self._initialize_progress_tracking()
    
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
    
    def _initialize_topic_difficulty(self) -> Dict[str, str]:
        """Initialize difficulty levels for each topic"""
        return {
            # Data Preprocessing - Beginner
            "Missing Data": "Beginner",
            "Encoding": "Beginner", 
            "Feature Scaling": "Beginner",
            
            # Regression - Beginner to Intermediate
            "Linear Regression": "Beginner",
            "Polynomial Regression": "Intermediate",
            "SVR": "Advanced",
            "Decision Tree Regression": "Intermediate",
            "Random Forest Regression": "Intermediate",
            
            # Classification - Beginner to Advanced
            "Logistic Regression": "Beginner",
            "K-NN": "Beginner",
            "SVM": "Advanced",
            "Naive Bayes": "Intermediate",
            "Decision Tree Classification": "Intermediate",
            "Random Forest Classification": "Intermediate",
            
            # Clustering - Intermediate
            "K-Means": "Intermediate",
            "Hierarchical Clustering": "Advanced",
            
            # Association Rule Learning - Advanced
            "Apriori": "Advanced",
            "Eclat": "Advanced",
            
            # Reinforcement Learning - Advanced
            "UCB": "Advanced",
            "Thompson Sampling": "Advanced",
            "Q-Learning": "Advanced",
            
            # NLP - Intermediate to Advanced
            "Bag-of-Words": "Intermediate",
            "TF-IDF": "Intermediate",
            "Word2Vec": "Advanced",
            "BERT": "Advanced",
            
            # Deep Learning - Advanced
            "ANN": "Advanced",
            "CNN": "Advanced",
            "RNN": "Advanced",
            "LSTM": "Advanced",
            
            # Dimensionality Reduction - Intermediate to Advanced
            "PCA": "Intermediate",
            "LDA": "Advanced",
            "Kernel PCA": "Advanced",
            
            # Model Selection & Boosting - Intermediate to Advanced
            "Cross Validation": "Intermediate",
            "Grid Search": "Intermediate",
            "XGBoost": "Advanced",
            "AdaBoost": "Advanced"
        }
    
    def _initialize_learning_paths(self) -> Dict[str, List[str]]:
        """Initialize recommended learning paths"""
        return {
            "Beginner Path": [
                "Missing Data", "Feature Scaling", "Linear Regression", 
                "Logistic Regression", "K-NN", "K-Means"
            ],
            "Data Science Path": [
                "Data Preprocessing", "Linear Regression", "Logistic Regression",
                "Random Forest Classification", "Cross Validation", "PCA"
            ],
            "Deep Learning Path": [
                "Linear Regression", "Logistic Regression", "ANN", 
                "CNN", "RNN", "LSTM"
            ],
            "NLP Specialist Path": [
                "Bag-of-Words", "TF-IDF", "Word2Vec", "BERT"
            ],
            "Advanced ML Path": [
                "SVM", "XGBoost", "Q-Learning", "BERT", "Kernel PCA"
            ]
        }
    
    def _initialize_prerequisites(self) -> Dict[str, List[str]]:
        """Initialize prerequisites for each topic"""
        return {
            "Polynomial Regression": ["Linear Regression"],
            "Random Forest Regression": ["Decision Tree Regression"],
            "Random Forest Classification": ["Decision Tree Classification"],
            "SVM": ["Logistic Regression"],
            "Hierarchical Clustering": ["K-Means"],
            "Word2Vec": ["Bag-of-Words", "TF-IDF"],
            "BERT": ["Word2Vec"],
            "CNN": ["ANN"],
            "RNN": ["ANN"],
            "LSTM": ["RNN"],
            "LDA": ["PCA"],
            "Kernel PCA": ["PCA"],
            "XGBoost": ["Decision Tree Classification"],
            "AdaBoost": ["Decision Tree Classification"],
            "Grid Search": ["Cross Validation"]
        }
    
    def _initialize_progress_tracking(self):
        """Initialize progress tracking in session state"""
        if 'topic_progress' not in st.session_state:
            st.session_state['topic_progress'] = {}
        if 'quiz_scores' not in st.session_state:
            st.session_state['quiz_scores'] = {}
        if 'completed_topics' not in st.session_state:
            st.session_state['completed_topics'] = set()
        if 'learning_streak' not in st.session_state:
            st.session_state['learning_streak'] = 0
        if 'last_activity' not in st.session_state:
            st.session_state['last_activity'] = None
    
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
    
    def get_topic_difficulty(self, topic: str) -> str:
        """Get difficulty level for a topic"""
        return self.topic_difficulty.get(topic, "Intermediate")
    
    def get_topics_by_difficulty(self, difficulty: str) -> List[str]:
        """Get all topics of a specific difficulty level"""
        return [topic for topic, level in self.topic_difficulty.items() if level == difficulty]
    
    def get_learning_paths(self) -> Dict[str, List[str]]:
        """Get all available learning paths"""
        return self.learning_paths
    
    def get_recommended_next_topics(self, completed_topics: List[str]) -> List[str]:
        """Get recommended next topics based on completed topics"""
        recommendations = []
        
        for topic, prerequisites in self.prerequisites.items():
            if topic not in completed_topics:
                if all(prereq in completed_topics for prereq in prerequisites):
                    recommendations.append(topic)
        
        # If no prerequisites-based recommendations, suggest beginner topics
        if not recommendations:
            beginner_topics = self.get_topics_by_difficulty("Beginner")
            recommendations = [topic for topic in beginner_topics if topic not in completed_topics]
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def update_topic_progress(self, topic: str, progress_type: str, score: Optional[float] = None):
        """Update progress for a topic"""
        if topic not in st.session_state['topic_progress']:
            st.session_state['topic_progress'][topic] = {
                'explained': False,
                'quiz_completed': False,
                'coding_completed': False,
                'dataset_completed': False,
                'last_accessed': None
            }
        
        # Update specific progress type
        if progress_type == 'explanation':
            st.session_state['topic_progress'][topic]['explained'] = True
        elif progress_type == 'quiz' and score is not None:
            st.session_state['topic_progress'][topic]['quiz_completed'] = True
            st.session_state['quiz_scores'][topic] = score
        elif progress_type == 'coding':
            st.session_state['topic_progress'][topic]['coding_completed'] = True
        elif progress_type == 'dataset':
            st.session_state['topic_progress'][topic]['dataset_completed'] = True
        
        # Update last accessed time
        st.session_state['topic_progress'][topic]['last_accessed'] = datetime.now()
        
        # Check if topic is completed (all activities done)
        progress = st.session_state['topic_progress'][topic]
        if all([progress['explained'], progress['quiz_completed'], progress['coding_completed']]):
            st.session_state['completed_topics'].add(topic)
            self._update_learning_streak()
    
    def _update_learning_streak(self):
        """Update learning streak"""
        today = datetime.now().date()
        last_activity = st.session_state.get('last_activity')
        
        if last_activity is None:
            st.session_state['learning_streak'] = 1
        elif last_activity == today:
            # Same day, don't increment
            pass
        elif (today - last_activity).days == 1:
            # Consecutive day, increment streak
            st.session_state['learning_streak'] += 1
        else:
            # Streak broken, reset
            st.session_state['learning_streak'] = 1
        
        st.session_state['last_activity'] = today
    
    def get_topic_progress(self, topic: str) -> Dict:
        """Get progress information for a topic"""
        return st.session_state['topic_progress'].get(topic, {
            'explained': False,
            'quiz_completed': False,
            'coding_completed': False,
            'dataset_completed': False,
            'last_accessed': None
        })
    
    def get_overall_progress(self) -> Dict:
        """Get overall learning progress statistics"""
        total_topics = sum(len(algorithms) for algorithms in self.curriculum.values())
        completed_topics = len(st.session_state['completed_topics'])
        
        # Calculate category-wise progress
        category_progress = {}
        for category, algorithms in self.curriculum.items():
            completed_in_category = sum(1 for algo in algorithms if algo in st.session_state['completed_topics'])
            category_progress[category] = {
                'completed': completed_in_category,
                'total': len(algorithms),
                'percentage': (completed_in_category / len(algorithms)) * 100 if algorithms else 0
            }
        
        # Calculate average quiz score
        quiz_scores = list(st.session_state['quiz_scores'].values())
        avg_quiz_score = sum(quiz_scores) / len(quiz_scores) if quiz_scores else 0
        
        return {
            'total_topics': total_topics,
            'completed_topics': completed_topics,
            'completion_percentage': (completed_topics / total_topics) * 100,
            'category_progress': category_progress,
            'learning_streak': st.session_state['learning_streak'],
            'average_quiz_score': avg_quiz_score,
            'topics_with_progress': len(st.session_state['topic_progress'])
        }
    
    def get_practice_problem_availability(self, topic: str) -> Dict[str, bool]:
        """Check which practice problem types are available for a topic"""
        # Topics with available practice problems (based on our implementation)
        available_topics = {
            "Missing Data", "Feature Scaling", "Linear Regression", "Random Forest Regression",
            "Logistic Regression", "K-NN", "Random Forest Classification", "K-Means", "PCA", "ANN"
        }
        
        has_problems = topic in available_topics
        
        return {
            'quiz': has_problems,
            'coding': has_problems,
            'dataset': has_problems
        }
    
    def get_curriculum_with_progress(self) -> Dict[str, List[Dict]]:
        """Get curriculum with progress information for each topic"""
        curriculum_with_progress = {}
        
        for category, algorithms in self.curriculum.items():
            algorithms_with_progress = []
            for algorithm in algorithms:
                progress = self.get_topic_progress(algorithm)
                difficulty = self.get_topic_difficulty(algorithm)
                practice_availability = self.get_practice_problem_availability(algorithm)
                
                algorithms_with_progress.append({
                    'name': algorithm,
                    'difficulty': difficulty,
                    'progress': progress,
                    'practice_available': practice_availability,
                    'completed': algorithm in st.session_state['completed_topics']
                })
            
            curriculum_with_progress[category] = algorithms_with_progress
        
        return curriculum_with_progress
