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
    
    def _setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="ML Algorithm Tutor",
            page_icon="🤖",
            layout="wide"
        )
    
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
            /* Adding styles for visualization section */
            .viz-container {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
                border: 2px solid #e9ecef;
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
                        self.memory_manager.set_selected_topic(algorithm)
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
    
    def render_visualization_section(self, topic: str):
        """Render interactive visualizations for the topic"""
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
        else:
            # Show algorithm comparison chart as fallback
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            comparison_fig = self.tutor.get_algorithm_comparison()
            st.plotly_chart(comparison_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.info("💡 **Algorithm Comparison**: While a specific visualization isn't available for this topic, "
                   "here's how different ML algorithms compare in terms of accuracy, speed, and interpretability.")
    
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
