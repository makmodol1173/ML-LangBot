"""
User Interface Module
Handles all Streamlit UI components and user interactions
"""

import streamlit as st
from .memory import MemoryManager
from .curriculum import CurriculumManager
from .tutor import MLTutor


class MLTutorUI:
    """Main UI class for the ML Algorithm Tutor"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.curriculum_manager = CurriculumManager()
        self.tutor = MLTutor()
        self._setup_page_config()
        self._load_custom_css()
        self._initialize_practice_state()
    
    def _setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="ML Algorithm Tutor",
            page_icon="🤖",
            layout="wide"
        )
    
    def _initialize_practice_state(self):
        """Initialize session state for practice problems"""
        if 'practice_mode' not in st.session_state:
            st.session_state['practice_mode'] = False
        if 'current_quiz' not in st.session_state:
            st.session_state['current_quiz'] = None
        if 'quiz_answers' not in st.session_state:
            st.session_state['quiz_answers'] = {}
        if 'quiz_submitted' not in st.session_state:
            st.session_state['quiz_submitted'] = False
        if 'coding_solution' not in st.session_state:
            st.session_state['coding_solution'] = ""
        if 'selected_practice_topic' not in st.session_state:
            st.session_state['selected_practice_topic'] = None
    
    def _load_custom_css(self):
        """Load custom CSS for better UI"""
        st.markdown("""
        <style>
            .main-header {
                font-size: 3rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
                font-weight: bold;
            }
            .sub-header {
                font-size: 1.5rem;
                color: #ff7f0e;
                margin-bottom: 1rem;
            }
            .algorithm-card {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border-left: 4px solid #1f77b4;
            }
            .chat-container {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 1rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .user-message {
                background-color: #e3f2fd;
                padding: 0.5rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                text-align: right;
            }
            .bot-message {
                background-color: #f5f5f5;
                padding: 0.5rem;
                border-radius: 10px;
                margin: 0.5rem 0;
            }
            .viz-container {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
                border: 2px solid #e9ecef;
            }
            /* Adding styles for practice problems section */
            .practice-container {
                background-color: #fff3cd;
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                border: 2px solid #ffeaa7;
            }
            .quiz-question {
                background-color: #e8f4fd;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                border-left: 4px solid #2196f3;
            }
            .quiz-option {
                background-color: #f8f9fa;
                padding: 0.5rem;
                margin: 0.3rem 0;
                border-radius: 5px;
                border: 1px solid #dee2e6;
                cursor: pointer;
            }
            .quiz-option:hover {
                background-color: #e9ecef;
            }
            .correct-answer {
                background-color: #d4edda !important;
                border-color: #28a745 !important;
            }
            .incorrect-answer {
                background-color: #f8d7da !important;
                border-color: #dc3545 !important;
            }
            .coding-problem {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                border-left: 4px solid #6c757d;
            }
            .practice-tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 1rem;
            }
            .practice-tab {
                padding: 0.5rem 1rem;
                border-radius: 5px;
                border: 1px solid #dee2e6;
                background-color: #f8f9fa;
                cursor: pointer;
            }
            .practice-tab.active {
                background-color: #007bff;
                color: white;
                border-color: #007bff;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🤖 ML Algorithm Tutor</h1>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the curriculum sidebar"""
        with st.sidebar:
            st.markdown('<h2 class="sub-header">📚 ML Curriculum</h2>', unsafe_allow_html=True)
            
            curriculum = self.curriculum_manager.get_curriculum()
            for category, algorithms in curriculum.items():
                st.markdown(f"**{category}**")
                for algorithm in algorithms:
                    unique_key = f"btn_{category}_{algorithm}".replace(" ", "_").replace("&", "and")
                    if st.button(f" {algorithm}", key=unique_key):
                        if st.session_state.get('practice_mode', False):
                            st.session_state['selected_practice_topic'] = algorithm
                        else:
                            self.memory_manager.set_selected_topic(algorithm)
                        st.rerun()
            
            st.markdown("---")
            st.markdown('<h3 class="sub-header">🎯 Practice Mode</h3>', unsafe_allow_html=True)
            
            practice_mode = st.toggle("Enable Practice Problems", value=st.session_state.get('practice_mode', False))
            if practice_mode != st.session_state.get('practice_mode', False):
                st.session_state['practice_mode'] = practice_mode
                if practice_mode:
                    st.session_state['selected_practice_topic'] = None
                else:
                    self.memory_manager.set_selected_topic(None)
                st.rerun()
    
    def render_input_section(self):
        """Render the topic input section"""
        st.markdown('<h2 class="sub-header">💬 Ask About ML</h2>', unsafe_allow_html=True)
        
        # Manual topic input
        topic_input = st.text_input(
            "Enter any ML algorithm or topic:",
            placeholder="e.g., Random Forest, K-Means, PCA..."
        )
        
        if st.button("🚀 Get Explanation", key="explain_btn"):
            if topic_input.strip():
                self.memory_manager.set_selected_topic(topic_input.strip())
                st.rerun()
            else:
                st.warning("Please enter a topic!")
        
        # Quick topic suggestions
        st.markdown("**💡 Quick Topics:**")
        quick_topics = self.curriculum_manager.get_quick_topics()
        for topic in quick_topics:
            if st.button(f"⚡ {topic}", key=f"quick_{topic}"):
                self.memory_manager.set_selected_topic(topic)
                st.rerun()
    
    def render_explanation_section(self):
        """Render the explanation section"""
        if st.session_state.get('practice_mode', False):
            self.render_practice_mode_section()
            return
            
        st.markdown('<h2 class="sub-header">📖 ML Explanation</h2>', unsafe_allow_html=True)
        
        selected_topic = self.memory_manager.get_selected_topic()
        
        if selected_topic:
            if not self.tutor.is_initialized():
                st.error("LLM not initialized. Please check your Google API key.")
                return
            
            with st.spinner(f"🤖 Generating explanation for {selected_topic}..."):
                explanation_placeholder = st.empty()
                chat_history = self.memory_manager.get_chat_history()
                
                explanation = ""
                for partial in self.tutor.get_explanation_stream(selected_topic, chat_history):
                    explanation = partial
                    explanation_placeholder.markdown(f"**Topic: {selected_topic}**\n\n{explanation}")
                
                if explanation and not explanation.startswith("Error"):
                    # Save to memory
                    self.memory_manager.add_message("user", f"Explain: {selected_topic}")
                    self.memory_manager.add_message("assistant", explanation)
                    self.memory_manager.save_conversation(f"Explain: {selected_topic}", explanation)
                    
                    self.render_visualization_section(selected_topic)
                else:
                    st.error(explanation)
    
    def render_practice_mode_section(self):
        """Render the practice mode section"""
        st.markdown('<h2 class="sub-header">🎯 Practice Problems</h2>', unsafe_allow_html=True)
        
        selected_topic = st.session_state.get('selected_practice_topic')
        
        if selected_topic:
            st.markdown(f"**Selected Topic: {selected_topic}**")
            self.render_practice_problems_section(selected_topic)
        else:
            st.info("👈 Select a topic from the sidebar to start practicing!")
    
    def render_practice_problems_section(self, topic: str):
        """Render practice problems section for the given topic"""
        st.markdown('<h3 class="sub-header">🎯 Practice Problems</h3>', unsafe_allow_html=True)
        
        # Practice problem tabs
        tab1, tab2, tab3 = st.tabs(["📝 Quiz", "💻 Coding", "📊 Dataset"])
        
        with tab1:
            self.render_quiz_section(topic)
        
        with tab2:
            self.render_coding_section(topic)
        
        with tab3:
            self.render_dataset_section(topic)
    
    def render_quiz_section(self, topic: str):
        """Render quiz questions section"""
        st.markdown('<div class="practice-container">', unsafe_allow_html=True)
        
        # Get quiz questions
        quiz_questions = self.tutor.get_quiz_questions(topic, 2)
        
        if not quiz_questions or (len(quiz_questions) == 1 and "not available" in quiz_questions[0]["question"]):
            st.info(f"Quiz questions for {topic} are not available yet. Try another topic!")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        st.markdown(f"**Quiz: {topic}**")
        
        # Display questions
        for i, question in enumerate(quiz_questions):
            st.markdown(f'<div class="quiz-question">', unsafe_allow_html=True)
            st.markdown(f"**Question {i+1}:** {question['question']}")
            
            # Multiple choice options
            options = question['options']
            selected_option = st.radio(
                "Choose your answer:",
                options,
                key=f"quiz_{topic}_{i}",
                index=None
            )
            
            # Store answer
            if selected_option:
                st.session_state['quiz_answers'][f"{topic}_{i}"] = options.index(selected_option)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Submit quiz button
        if st.button("Submit Quiz", key=f"submit_quiz_{topic}"):
            self.evaluate_quiz(topic, quiz_questions)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def evaluate_quiz(self, topic: str, questions: list):
        """Evaluate quiz answers and show results"""
        st.markdown("### Quiz Results")
        
        total_questions = len(questions)
        correct_answers = 0
        
        for i, question in enumerate(questions):
            user_answer = st.session_state['quiz_answers'].get(f"{topic}_{i}")
            correct_answer = question['correct']
            
            if user_answer is not None:
                if user_answer == correct_answer:
                    correct_answers += 1
                    st.success(f"Question {i+1}: Correct! ✅")
                else:
                    st.error(f"Question {i+1}: Incorrect ❌")
                    st.info(f"Correct answer: {question['options'][correct_answer]}")
                
                st.markdown(f"**Explanation:** {question['explanation']}")
            else:
                st.warning(f"Question {i+1}: Not answered")
        
        # Overall score
        score = (correct_answers / total_questions) * 100
        st.markdown(f"### Overall Score: {score:.1f}% ({correct_answers}/{total_questions})")
        
        if score >= 80:
            st.balloons()
            st.success("Excellent work! You have a strong understanding of this topic.")
        elif score >= 60:
            st.success("Good job! Consider reviewing the explanations for better understanding.")
        else:
            st.info("Keep practicing! Review the topic explanation and try again.")
    
    def render_coding_section(self, topic: str):
        """Render coding problems section"""
        st.markdown('<div class="practice-container">', unsafe_allow_html=True)
        
        # Get coding problem
        coding_problem = self.tutor.get_coding_problem(topic)
        
        if not coding_problem or "not available" in coding_problem.get("problem", ""):
            st.info(f"Coding problems for {topic} are not available yet. Try another topic!")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        st.markdown(f"**Coding Challenge: {topic}**")
        st.markdown(f'<div class="coding-problem">', unsafe_allow_html=True)
        st.markdown(f"**Problem:** {coding_problem['problem']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show starter code
        st.markdown("**Starter Code:**")
        st.code(coding_problem['starter_code'], language='python')
        
        # Code input area
        st.markdown("**Your Solution:**")
        user_code = st.text_area(
            "Write your code here:",
            value=st.session_state.get('coding_solution', ''),
            height=200,
            key=f"code_input_{topic}"
        )
        
        # Update session state
        st.session_state['coding_solution'] = user_code
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Show Solution", key=f"show_solution_{topic}"):
                st.markdown("**Expected Solution:**")
                st.code(coding_problem['solution'], language='python')
                st.info(f"**Explanation:** {coding_problem['explanation']}")
        
        with col2:
            if st.button("Evaluate My Code", key=f"evaluate_code_{topic}"):
                if user_code.strip():
                    with st.spinner("Evaluating your code..."):
                        evaluation = self.tutor.evaluate_coding_solution(topic, user_code)
                        st.markdown("**AI Evaluation:**")
                        st.markdown(evaluation)
                else:
                    st.warning("Please write some code first!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_dataset_section(self, topic: str):
        """Render dataset problems section"""
        st.markdown('<div class="practice-container">', unsafe_allow_html=True)
        
        # Get dataset problem
        dataset_problem = self.tutor.get_dataset_problem(topic)
        
        if dataset_problem['dataset'].empty:
            st.info(f"Dataset problems for {topic} are not available yet. Try another topic!")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        st.markdown(f"**Dataset Challenge: {topic}**")
        st.markdown(f"**Description:** {dataset_problem['description']}")
        
        # Show dataset
        st.markdown("**Dataset Preview:**")
        st.dataframe(dataset_problem['dataset'].head(10))
        
        # Dataset info
        st.markdown("**Dataset Information:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"- **Rows:** {len(dataset_problem['dataset'])}")
            st.markdown(f"- **Features:** {', '.join(dataset_problem['features'])}")
        
        with col2:
            st.markdown(f"- **Target:** {dataset_problem['target']}")
            st.markdown(f"- **Task Type:** {dataset_problem['task_type']}")
        
        # Download dataset button
        csv = dataset_problem['dataset'].to_csv(index=False)
        st.download_button(
            label="Download Dataset (CSV)",
            data=csv,
            file_name=f"{topic.lower().replace(' ', '_')}_dataset.csv",
            mime="text/csv"
        )
        
        st.info("💡 **Challenge:** Use this dataset to practice implementing the algorithm. "
               "Try different parameters and compare results!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_visualization_section(self, topic: str):
        """Render interactive visualizations for the topic"""
        if not self.tutor.has_visualization(topic):
            return  # Don't show visualization section at all
        
        st.markdown('<h3 class="sub-header">📊 Interactive Visualization</h3>', unsafe_allow_html=True)
        
        # Get visualization for the topic
        viz_fig = self.tutor.get_visualization(topic)
        
        if viz_fig:
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            st.plotly_chart(viz_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add explanation for the visualization
            st.info(f"💡 **Visualization Insight**: This interactive chart shows how {topic} works with sample data. "
                   f"You can hover over points for details and zoom in/out to explore the algorithm's behavior.")
    
    def render_chat_history(self):
        """Render the chat history section"""
        messages = self.memory_manager.get_messages()
        
        if messages:
            st.markdown('<h3 class="sub-header">💭 Chat History</h3>', unsafe_allow_html=True)
            chat_container = st.container()
            
            with chat_container:
                for message in messages:
                    if message["role"] == "user":
                        st.markdown(
                            f'<div class="user-message">👤 {message["content"]}</div>', 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="bot-message">🤖 {message["content"]}</div>', 
                            unsafe_allow_html=True
                        )
        
        # Clear chat button
        if messages:
            if st.button("🗑️ Clear Chat History"):
                self.memory_manager.clear_history()
                st.rerun()
    
    def run(self):
        """Main method to run the UI"""
        self.render_header()
        self.render_sidebar()
        
        # Main content area with better proportions for visualizations
        col1, col2 = st.columns([1, 3])
        
        with col1:
            self.render_input_section()
        
        with col2:
            self.render_explanation_section()
            self.render_chat_history()
