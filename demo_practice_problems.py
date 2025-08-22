"""
Demo Script for Practice Problems Feature
Demonstrates the complete practice problems functionality
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.practice_problems import PracticeProblemsGenerator
from app.tutor import MLTutor
from app.curriculum import CurriculumManager

# Load environment variables
load_dotenv()


def demo_practice_problems():
    """Demonstrate practice problems functionality"""
    print("🎯 ML Algorithm Tutor - Practice Problems Demo")
    print("=" * 50)
    
    # Initialize components
    generator = PracticeProblemsGenerator()
    tutor = MLTutor()
    curriculum = CurriculumManager()
    
    # Demo topic
    demo_topic = "Linear Regression"
    
    print(f"\n📚 Demonstrating Practice Problems for: {demo_topic}")
    print("-" * 50)
    
    # 1. Show topic information
    difficulty = curriculum.get_topic_difficulty(demo_topic)
    print(f"🎚️  Difficulty Level: {difficulty}")
    
    availability = curriculum.get_practice_problem_availability(demo_topic)
    print(f"📋 Available Practice Types: {', '.join([k for k, v in availability.items() if v])}")
    
    # 2. Demo Quiz Questions
    print(f"\n📝 QUIZ QUESTIONS")
    print("-" * 30)
    
    quiz_questions = generator.get_quiz_questions(demo_topic, 2)
    for i, question in enumerate(quiz_questions, 1):
        print(f"\nQuestion {i}: {question['question']}")
        for j, option in enumerate(question['options']):
            marker = "→" if j == question['correct'] else " "
            print(f"  {marker} {chr(65+j)}. {option}")
        print(f"💡 Explanation: {question['explanation']}")
    
    # 3. Demo Coding Problem
    print(f"\n💻 CODING CHALLENGE")
    print("-" * 30)
    
    coding_problem = generator.generate_coding_problem(demo_topic)
    print(f"Problem: {coding_problem['problem']}")
    print(f"\nStarter Code:")
    print(coding_problem['starter_code'])
    print(f"\n💡 Solution Approach: {coding_problem['explanation']}")
    
    # 4. Demo Dataset Problem
    print(f"\n📊 DATASET CHALLENGE")
    print("-" * 30)
    
    dataset_problem = generator.generate_dataset_problem(demo_topic)
    print(f"Description: {dataset_problem['description']}")
    print(f"Dataset Shape: {dataset_problem['dataset'].shape}")
    print(f"Features: {', '.join(dataset_problem['features'])}")
    print(f"Target: {dataset_problem['target']}")
    
    # 5. Demo Learning Paths
    print(f"\n🛤️  LEARNING PATHS")
    print("-" * 30)
    
    learning_paths = curriculum.get_learning_paths()
    for path_name, topics in learning_paths.items():
        if demo_topic in topics:
            print(f"📍 {path_name}: {' → '.join(topics)}")
    
    # 6. Demo Progress Tracking
    print(f"\n📈 PROGRESS TRACKING")
    print("-" * 30)
    
    # Simulate progress updates
    curriculum.update_topic_progress(demo_topic, 'explanation')
    curriculum.update_topic_progress(demo_topic, 'quiz', 85.0)
    curriculum.update_topic_progress(demo_topic, 'coding')
    
    progress = curriculum.get_topic_progress(demo_topic)
    overall_progress = curriculum.get_overall_progress()
    
    print(f"Topic Progress:")
    print(f"  ✅ Explanation: {progress['explained']}")
    print(f"  ✅ Quiz: {progress['quiz_completed']} (Score: 85%)")
    print(f"  ✅ Coding: {progress['coding_completed']}")
    print(f"  ⏳ Dataset: {progress['dataset_completed']}")
    
    print(f"\nOverall Statistics:")
    print(f"  📊 Completion: {overall_progress['completion_percentage']:.1f}%")
    print(f"  🔥 Learning Streak: {overall_progress['learning_streak']} days")
    print(f"  📝 Average Quiz Score: {overall_progress['average_quiz_score']:.1f}%")
    
    # 7. Demo AI Integration (if available)
    if tutor.is_initialized():
        print(f"\n🤖 AI-POWERED FEATURES")
        print("-" * 30)
        
        print("Generating custom practice problem...")
        custom_problem = tutor.generate_custom_practice_problem(demo_topic, difficulty, "scenario")
        print(f"Custom Problem: {custom_problem[:200]}...")
        
        print("\nEvaluating sample code...")
        sample_code = "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()"
        evaluation = tutor.evaluate_coding_solution(demo_topic, sample_code)
        print(f"AI Evaluation: {evaluation[:150]}...")
    else:
        print(f"\n⚠️  AI features require GOOGLE_API_KEY environment variable")
    
    print(f"\n🎉 Demo Complete!")
    print("To try the full interactive experience, run: python run_app.py")


if __name__ == "__main__":
    demo_practice_problems()
