import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

class MLTutor:
    """Handles AI logic and ML explanations"""
    
    def __init__(self):
        load_dotenv()
        self.llm = None
        self.prompt_template = self._create_prompt_template()
        self._initialize_llm()
    
    def _create_prompt_template(self):
        """Create the ML tutor prompt template"""
        return PromptTemplate(
            input_variables=["topic", "chat_history"],
            template="""You are a friendly and expert Machine Learning tutor and code assistant.

Your task is to answer only about machine learning algorithms or libraries.

When given a topic (algorithm or library), respond strictly in this format:

---

Overview:
- A brief introduction (2–3 sentences) explaining what it is and why it is used.

Explanation:
- A clear, concise explanation of how it works, its key concepts, and typical use cases.
- Keep it simple, informative, and easy to understand.

Snippet Code:
# A short, complete, and runnable Python example
# Use commonly available libraries like scikit-learn, pandas, numpy, tensorflow, or pytorch
# Keep code minimal, clean, and readable

---

Rules:
1. Respond only in the above structure — do not add extra text.
2. If the topic is not a valid ML algorithm or library, politely refuse and ask for a proper ML topic.
3. Keep explanations concise, code clean, and readable.

Topic: {topic}

Chat History: {chat_history}

Please provide your response in the exact format specified above."""
        )
    
    def _initialize_llm(self):
        """Initialize the Gemini language model"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.7,
                google_api_key=api_key
            )
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            self.llm = None
    
    def is_initialized(self):
        """Check if LLM is properly initialized"""
        return self.llm is not None
    
    def get_explanation(self, topic, chat_history=""):
        """Get ML explanation from the chatbot"""
        if not self.llm:
            return "Error: LLM not initialized. Please check your Google API key."
        
        try:
            prompt_text = self.prompt_template.format(topic=topic, chat_history=chat_history)
            response = self.llm.invoke(prompt_text)
            return getattr(response, 'content', str(response))
        except Exception as e:
            return f"Error getting explanation: {e}"
    
    def get_explanation_stream(self, topic, chat_history=""):
        """Stream ML explanation from the chatbot"""
        if not self.llm:
            yield "Error: LLM not initialized. Please check your Google API key."
            return
        
        try:
            prompt_text = self.prompt_template.format(topic=topic, chat_history=chat_history)
            response_stream = self.llm.stream(prompt_text)
            explanation = ""
            for chunk in response_stream:
                content = getattr(chunk, 'content', str(chunk))
                explanation += content
                yield explanation
        except Exception as e:
            yield f"Error getting explanation: {e}"
