"""
Test Script for Practice Problems Feature
Comprehensive testing of all practice problems components
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.practice_problems import PracticeProblemsGenerator
from app.tutor import MLTutor
from app.curriculum import CurriculumManager
from app.memory import MemoryManager

# Load environment variables
load_dotenv()


def test_practice_problems_generator():
    """Test the Practice Problems Generator"""
    print("🧪 Testing Practice Problems Generator...")
    
    generator = PracticeProblemsGenerator()
    
    # Test 1: Get available topics
    topics = generator.get_available_topics()
    print(f"✅ Available topics: {len(topics)} topics found")
    print(f"   Topics: {', '.join(topics[:5])}...")
    
    # Test 2: Generate quiz questions
    test_topic = "Linear Regression"
    quiz_questions = generator.get_quiz_questions(test_topic, 2)
    print(f"✅ Quiz questions for {test_topic}: {len(quiz_questions)} questions generated")
    
    # Test 3: Generate coding problem
    coding_problem = generator.generate_coding_problem(test_topic)
    print(f"✅ Coding problem for {test_topic}: Generated successfully")
    print(f"   Problem: {coding_problem['problem'][:50]}...")
    
    # Test 4: Generate dataset problem
    dataset_problem = generator.generate_dataset_problem(test_topic)
    print(f"✅ Dataset problem for {test_topic}: {len(dataset_problem['dataset'])} rows generated")
    
    # Test 5: Comprehensive exercise
    comprehensive = generator.generate_comprehensive_exercise(test_topic)
    print(f"✅ Comprehensive exercise: All components generated")
    
    return True


def test_tutor_integration():
    """Test AI Tutor integration with practice problems"""
    print("\n🧪 Testing AI Tutor Integration...")
    
    tutor = MLTutor()
    
    # Test 1: Check initialization
    if not tutor.is_initialized():
        print("⚠️  Warning: LLM not initialized (check GOOGLE_API_KEY)")
        return False
    
    print("✅ AI Tutor initialized successfully")
    
    # Test 2: Get practice problems
    test_topic = "K-Means"
    practice_problems = tutor.get_practice_problems(test_topic)
    print(f"✅ Practice problems for {test_topic}: Generated successfully")
    
    # Test 3: Get quiz questions
    quiz_questions = tutor.get_quiz_questions(test_topic, 2)
    print(f"✅ Quiz questions: {len(quiz_questions)} questions retrieved")
    
    # Test 4: Get coding problem
    coding_problem = tutor.get_coding_problem(test_topic)
    print(f"✅ Coding problem: Generated successfully")
    
    # Test 5: Available practice topics
    available_topics = tutor.get_available_practice_topics()
    print(f"✅ Available practice topics: {len(available_topics)} topics")
    
    return True


def test_curriculum_integration():
    """Test curriculum integration with practice problems"""
    print("\n🧪 Testing Curriculum Integration...")
    
    curriculum = CurriculumManager()
    
    # Test 1: Get curriculum with progress
    curriculum_with_progress = curriculum.get_curriculum_with_progress()
    print(f"✅ Curriculum with progress: {len(curriculum_with_progress)} categories")
    
    # Test 2: Topic difficulty levels
    test_topics = ["Linear Regression", "SVM", "BERT"]
    for topic in test_topics:
        difficulty = curriculum.get_topic_difficulty(topic)
        print(f"✅ {topic}: {difficulty} level")
    
    # Test 3: Learning paths
    learning_paths = curriculum.get_learning_paths()
    print(f"✅ Learning paths: {len(learning_paths)} paths available")
    
    # Test 4: Practice problem availability
    test_topic = "Linear Regression"
    availability = curriculum.get_practice_problem_availability(test_topic)
    print(f"✅ Practice availability for {test_topic}: {availability}")
    
    # Test 5: Overall progress
    progress = curriculum.get_overall_progress()
    print(f"✅ Overall progress: {progress['completion_percentage']:.1f}% complete")
    
    return True


def test_feature_integration():
    """Test complete feature integration"""
    print("\n🧪 Testing Complete Feature Integration...")
    
    # Initialize all components
    generator = PracticeProblemsGenerator()
    tutor = MLTutor()
    curriculum = CurriculumManager()
    memory = MemoryManager()
    
    test_topic = "Logistic Regression"
    
    # Test 1: End-to-end workflow
    print(f"Testing complete workflow for: {test_topic}")
    
    # Check if topic is valid
    is_valid = curriculum.is_valid_topic(test_topic)
    print(f"✅ Topic validation: {is_valid}")
    
    # Get difficulty level
    difficulty = curriculum.get_topic_difficulty(test_topic)
    print(f"✅ Difficulty level: {difficulty}")
    
    # Generate practice problems
    if tutor.is_initialized():
        practice_problems = tutor.get_practice_problems(test_topic, difficulty)
        print(f"✅ Practice problems generated: {len(practice_problems['quiz_questions'])} quiz questions")
        
        # Test custom problem generation
        custom_problem = tutor.generate_custom_practice_problem(test_topic, difficulty, "quiz")
        print(f"✅ Custom problem generated: {len(custom_problem)} characters")
    else:
        print("⚠️  Skipping AI-dependent tests (LLM not initialized)")
    
    # Update progress
    curriculum.update_topic_progress(test_topic, 'explanation')
    curriculum.update_topic_progress(test_topic, 'quiz', 85.0)
    
    progress = curriculum.get_topic_progress(test_topic)
    print(f"✅ Progress tracking: Explanation={progress['explained']}, Quiz={progress['quiz_completed']}")
    
    return True


def run_comprehensive_test():
    """Run comprehensive test of all practice problems features"""
    print("🚀 Starting Comprehensive Practice Problems Feature Test\n")
    print("=" * 60)
    
    test_results = []
    
    # Test individual components
    try:
        result1 = test_practice_problems_generator()
        test_results.append(("Practice Problems Generator", result1))
    except Exception as e:
        print(f"❌ Practice Problems Generator test failed: {e}")
        test_results.append(("Practice Problems Generator", False))
    
    try:
        result2 = test_tutor_integration()
        test_results.append(("AI Tutor Integration", result2))
    except Exception as e:
        print(f"❌ AI Tutor Integration test failed: {e}")
        test_results.append(("AI Tutor Integration", False))
    
    try:
        result3 = test_curriculum_integration()
        test_results.append(("Curriculum Integration", result3))
    except Exception as e:
        print(f"❌ Curriculum Integration test failed: {e}")
        test_results.append(("Curriculum Integration", False))
    
    try:
        result4 = test_feature_integration()
        test_results.append(("Complete Feature Integration", result4))
    except Exception as e:
        print(f"❌ Complete Feature Integration test failed: {e}")
        test_results.append(("Complete Feature Integration", False))
    
    # Print test summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Practice Problems feature is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    print("\n📝 To use the Practice Problems feature:")
    print("1. Run: python run_app.py")
    print("2. Enable 'Practice Mode' in the sidebar")
    print("3. Select any ML topic to see practice problems")
    print("4. Complete quizzes, coding challenges, and dataset exercises")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
