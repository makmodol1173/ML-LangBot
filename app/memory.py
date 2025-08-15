"""
Memory Management Module
Handles chat history, session state, and conversation memory
"""

import streamlit as st
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Any


class MemoryManager:
    """Manages chat history and conversation memory for the ML tutor"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        
        if 'memory' not in st.session_state:
            st.session_state['memory'] = ConversationBufferMemory(return_messages=True)
        
        if 'selected_topic' not in st.session_state:
            st.session_state['selected_topic'] = None
    
    def add_message(self, role: str, content: str):
        """Add a message to the chat history"""
        st.session_state['messages'].append({
            "role": role,
            "content": content
        })
    
    def save_conversation(self, user_input: str, bot_response: str):
        """Save conversation to memory"""
        if 'memory' in st.session_state:
            st.session_state['memory'].save_context(
                {"input": user_input},
                {"output": bot_response}
            )
    
    def get_chat_history(self) -> str:
        """Get formatted chat history"""
        if 'memory' in st.session_state:
            return st.session_state['memory'].buffer
        return ""
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all chat messages"""
        return st.session_state.get('messages', [])
    
    def clear_history(self):
        """Clear chat history and memory"""
        st.session_state['messages'] = []
        if 'memory' in st.session_state:
            st.session_state['memory'].clear()
    
    def set_selected_topic(self, topic: str):
        """Set the currently selected topic"""
        st.session_state['selected_topic'] = topic
    
    def get_selected_topic(self) -> str:
        """Get the currently selected topic"""
        return st.session_state.get('selected_topic', '')
