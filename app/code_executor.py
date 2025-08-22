import sys
import io
import traceback
import contextlib
import ast
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Dict, Any, List, Tuple


class CodeExecutor:
    """Executes and evaluates user code safely"""
    
    def __init__(self):
        self.allowed_imports = {
            'numpy', 'np', 'pandas', 'pd', 'matplotlib', 'plt', 'seaborn', 'sns',
            'sklearn', 'scipy', 'math', 'random', 'collections', 'itertools',
            'warnings', 'os', 'sys'
        }
        self.test_cases = self._initialize_test_cases()
    
    def execute_code(self, code: str, test_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute user code and return results"""
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'variables': {},
            'test_results': []
        }
        
        try:
            # Redirect output
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Create execution environment
            exec_globals = {
                '__builtins__': __builtins__,
                'np': np,
                'pd': pd,
                'accuracy_score': accuracy_score,
                'mean_squared_error': mean_squared_error
            }
            
            # Add test data if provided
            if test_data:
                exec_globals.update(test_data)
            
            # Execute the code
            exec(code, exec_globals)
            
            # Capture variables
            result['variables'] = {k: v for k, v in exec_globals.items() 
                                 if not k.startswith('__') and not callable(v)}
            
            result['success'] = True
            result['output'] = stdout_capture.getvalue()
            
        except Exception as e:
            result['error'] = f"{type(e).__name__}: {str(e)}"
            result['output'] = stdout_capture.getvalue()
            
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        return result
    
    def validate_code_structure(self, code: str) -> Dict[str, Any]:
        """Validate code structure and provide feedback"""
        feedback = {
            'valid': True,
            'suggestions': [],
            'warnings': [],
            'score': 100
        }
        
        try:
            tree = ast.parse(code)
            
            # Check for required elements
            has_imports = any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
            has_variables = any(isinstance(node, ast.Assign) for node in ast.walk(tree))
            has_function_calls = any(isinstance(node, ast.Call) for node in ast.walk(tree))
            
            if not has_imports:
                feedback['suggestions'].append("Consider adding necessary imports")
                feedback['score'] -= 10
            
            if not has_variables:
                feedback['suggestions'].append("Create variables to store your results")
                feedback['score'] -= 15
            
            if not has_function_calls:
                feedback['suggestions'].append("Use appropriate functions to solve the problem")
                feedback['score'] -= 20
            
            # Check for best practices
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in ['X', 'y', 'model']:
                    feedback['suggestions'].append("Good use of standard ML variable names")
                    break
            
        except SyntaxError as e:
            feedback['valid'] = False
            feedback['warnings'].append(f"Syntax Error: {e.msg}")
            feedback['score'] = 0
        
        return feedback
    
    def run_test_cases(self, code: str, problem_type: str) -> List[Dict[str, Any]]:
        """Run predefined test cases for the code"""
        if problem_type not in self.test_cases:
            return [{'name': 'No tests available', 'passed': True, 'message': 'Manual review required'}]
        
        test_results = []
        test_suite = self.test_cases[problem_type]
        
        for test_case in test_suite:
            try:
                # Execute code with test data
                result = self.execute_code(code, test_case['input'])
                
                if result['success']:
                    # Check if expected variables exist
                    passed = True
                    message = "Test passed"
                    
                    for var_name, expected_value in test_case['expected'].items():
                        if var_name not in result['variables']:
                            passed = False
                            message = f"Variable '{var_name}' not found"
                            break
                        
                        actual_value = result['variables'][var_name]
                        if not self._compare_values(actual_value, expected_value):
                            passed = False
                            message = f"Expected {var_name}={expected_value}, got {actual_value}"
                            break
                    
                    test_results.append({
                        'name': test_case['name'],
                        'passed': passed,
                        'message': message
                    })
                else:
                    test_results.append({
                        'name': test_case['name'],
                        'passed': False,
                        'message': f"Execution error: {result['error']}"
                    })
                    
            except Exception as e:
                test_results.append({
                    'name': test_case['name'],
                    'passed': False,
                    'message': f"Test error: {str(e)}"
                })
        
        return test_results
    
    def _compare_values(self, actual, expected, tolerance=1e-6):
        """Compare values with tolerance for floating point numbers"""
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(actual - expected) < tolerance
        elif isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
            return np.allclose(actual, expected, atol=tolerance)
        else:
            return actual == expected
    
    def _initialize_test_cases(self) -> Dict[str, List[Dict]]:
        """Initialize test cases for different problem types"""
        return {
            'linear_regression': [
                {
                    'name': 'Basic Linear Regression',
                    'input': {
                        'X_train': np.array([[1], [2], [3], [4]]),
                        'y_train': np.array([2, 4, 6, 8]),
                        'X_test': np.array([[5]]),
                    },
                    'expected': {
                        'y_pred': np.array([10.0])  # Expected prediction for X=5
                    }
                }
            ],
            
            'classification': [
                {
                    'name': 'Binary Classification',
                    'input': {
                        'X_train': np.array([[1, 2], [2, 3], [3, 1], [4, 2]]),
                        'y_train': np.array([0, 0, 1, 1]),
                        'X_test': np.array([[2, 2], [3, 3]]),
                    },
                    'expected': {
                        'accuracy': 0.5  # Minimum expected accuracy
                    }
                }
            ],
            
            'clustering': [
                {
                    'name': 'K-Means Clustering',
                    'input': {
                        'X': np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]),
                        'n_clusters': 2
                    },
                    'expected': {
                        'n_clusters_': 2  # Expected number of clusters
                    }
                }
            ]
        }


class InteractiveCodeEditor:
    """Provides interactive code editing functionality"""
    
    def __init__(self):
        self.executor = CodeExecutor()
        self.code_templates = self._initialize_templates()
    
    def get_code_template(self, problem_type: str, difficulty: str = "Beginner") -> str:
        """Get code template for a specific problem type"""
        templates = self.code_templates.get(problem_type, {})
        return templates.get(difficulty, "# Start coding here...")
    
    def provide_hints(self, code: str, problem_type: str) -> List[str]:
        """Provide coding hints based on current code"""
        hints = []
        
        # Basic hints based on code analysis
        if 'import' not in code:
            hints.append("💡 Don't forget to import necessary libraries (numpy, sklearn, etc.)")
        
        if 'fit(' not in code and problem_type in ['linear_regression', 'classification']:
            hints.append("💡 Remember to train your model using the fit() method")
        
        if 'predict(' not in code and problem_type in ['linear_regression', 'classification']:
            hints.append("💡 Use predict() method to make predictions on test data")
        
        if problem_type == 'linear_regression' and 'LinearRegression' not in code:
            hints.append("💡 Try using LinearRegression from sklearn.linear_model")
        
        if problem_type == 'classification' and 'LogisticRegression' not in code:
            hints.append("💡 Consider using LogisticRegression for classification tasks")
        
        if problem_type == 'clustering' and 'KMeans' not in code:
            hints.append("💡 Use KMeans from sklearn.cluster for clustering")
        
        return hints[:3]  # Limit to 3 hints
    
    def evaluate_solution(self, code: str, problem_type: str) -> Dict[str, Any]:
        """Comprehensive evaluation of user solution"""
        # Execute code
        execution_result = self.executor.execute_code(code)
        
        # Validate structure
        structure_feedback = self.executor.validate_code_structure(code)
        
        # Run test cases
        test_results = self.executor.run_test_cases(code, problem_type)
        
        # Calculate overall score
        execution_score = 50 if execution_result['success'] else 0
        structure_score = structure_feedback['score'] * 0.3
        test_score = (sum(1 for test in test_results if test['passed']) / len(test_results)) * 50 if test_results else 0
        
        overall_score = execution_score + structure_score + test_score
        
        return {
            'execution': execution_result,
            'structure': structure_feedback,
            'tests': test_results,
            'score': min(100, overall_score),
            'feedback': self._generate_feedback(execution_result, structure_feedback, test_results)
        }
    
    def _generate_feedback(self, execution, structure, tests) -> str:
        """Generate comprehensive feedback for the user"""
        feedback_parts = []
        
        if execution['success']:
            feedback_parts.append("✅ Code executed successfully!")
        else:
            feedback_parts.append(f"❌ Execution failed: {execution['error']}")
        
        if structure['valid']:
            feedback_parts.append(f"📊 Code structure score: {structure['score']}/100")
        else:
            feedback_parts.append("⚠️ Code has syntax errors")
        
        passed_tests = sum(1 for test in tests if test['passed'])
        total_tests = len(tests)
        feedback_parts.append(f"🧪 Tests passed: {passed_tests}/{total_tests}")
        
        if structure['suggestions']:
            feedback_parts.append("💡 Suggestions:")
            feedback_parts.extend([f"  • {suggestion}" for suggestion in structure['suggestions']])
        
        return "\n".join(feedback_parts)
    
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize code templates for different problems"""
        return {
            'linear_regression': {
                'Beginner': '''# Linear Regression Template
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Your data is already loaded as X_train, y_train, X_test
# TODO: Create and train your model
model = LinearRegression()

# TODO: Fit the model

# TODO: Make predictions

# TODO: Calculate and print RMSE
''',
                'Intermediate': '''# Advanced Linear Regression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# TODO: Scale your features
# TODO: Train the model
# TODO: Evaluate with multiple metrics
''',
            },
            
            'classification': {
                'Beginner': '''# Classification Template
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Your data is already loaded as X_train, y_train, X_test, y_test
# TODO: Create and train your classifier

# TODO: Make predictions

# TODO: Calculate accuracy
''',
            },
            
            'clustering': {
                'Beginner': '''# K-Means Clustering Template
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Your data is already loaded as X
# TODO: Apply K-Means clustering

# TODO: Get cluster labels

# TODO: Visualize results (optional)
''',
            }
        }
