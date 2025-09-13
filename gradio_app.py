import gradio as gr
import os
import json
from datetime import datetime
from typing import List, Tuple, Optional
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

# Import your agent
from main import run, Agent, schema

class GradioAgentInterface:
    def __init__(self):
        self.conversation_history = []
        self.session_stats = {
            "total_queries": 0,
            "successful_responses": 0,
            "tools_used": 0,
            "images_generated": 0,
            "session_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    def process_query(self, message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str, str, str]:
        """Process user query and return updated history and stats"""
        if not message.strip():
            return history, "", self.get_stats_display(), ""
        
        self.session_stats["total_queries"] += 1
        
        try:
            # Run the agent
            result = run(message)
            
            if result["success"]:
                response = result["final_response"]
                self.session_stats["successful_responses"] += 1
                
                if result.get("tool_used", False):
                    self.session_stats["tools_used"] += 1
                
                # Check if image was generated (simple check)
                if "image" in message.lower() and "generate" in message.lower():
                    self.session_stats["images_generated"] += 1
                
                # Add to history
                history.append((message, response))
                
                # Create status message
                status = f"✅ Query processed successfully"
                if result.get("tool_used"):
                    status += " | 🛠️ Tools used"
                if result.get("retry_count", 0) > 0:
                    status += f" | 🔄 Retries: {result['retry_count']}"
                
                return history, "", self.get_stats_display(), status
            else:
                error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
                history.append((message, error_msg))
                return history, "", self.get_stats_display(), error_msg
                
        except Exception as e:
            error_msg = f"❌ System Error: {str(e)}"
            history.append((message, error_msg))
            return history, "", self.get_stats_display(), error_msg
    
    def get_stats_display(self) -> str:
        """Generate statistics display"""
        success_rate = (self.session_stats["successful_responses"] / max(1, self.session_stats["total_queries"])) * 100
        
        return f"""
**Session Statistics**
- Session Start: {self.session_stats['session_start']}
- Total Queries: {self.session_stats['total_queries']}
- Successful Responses: {self.session_stats['successful_responses']}
- Success Rate: {success_rate:.1f}%
- Tools Used: {self.session_stats['tools_used']}
- Images Generated: {self.session_stats['images_generated']}
"""
    
    def clear_conversation(self) -> Tuple[List, str, str]:
        """Clear conversation and reset stats"""
        self.conversation_history = []
        self.session_stats = {
            "total_queries": 0,
            "successful_responses": 0,
            "tools_used": 0,
            "images_generated": 0,
            "session_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return [], self.get_stats_display(), "🔄 Conversation cleared"
    
    def export_conversation(self, history: List[Tuple[str, str]]) -> str:
        """Export conversation to file"""
        if not history:
            return "No conversation to export"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_export_{timestamp}.txt"
        
        try:
            os.makedirs("exports", exist_ok=True)
            filepath = os.path.join("exports", filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Conversation Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                
                for i, (user_msg, bot_msg) in enumerate(history, 1):
                    f.write(f"Exchange {i}:\n")
                    f.write(f"User: {user_msg}\n")
                    f.write(f"Assistant: {bot_msg}\n")
                    f.write("-" * 40 + "\n\n")
                
                f.write("\nSession Statistics:\n")
                f.write(self.get_stats_display())
            
            return f"✅ Conversation exported to: {filepath}"
        except Exception as e:
            return f"❌ Export failed: {str(e)}"

def create_interface():
    """Create and configure the Gradio interface"""
    
    agent_interface = GradioAgentInterface()
    
    # Custom CSS for modern styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .header-text {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle-text {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 20px;
    }
    """

    with gr.Blocks(
        css=custom_css, 
        title="🤖 Multimodal AI Agent",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="gray"
        )
    ) as demo:
        
        # Header
        gr.HTML("""
            <div class="header-text">🤖 Multimodal AI Agent</div>
            <div class="subtitle-text">Powered by LangGraph • OpenAI • Hugging Face</div>
        """)

        with gr.Accordion("🔑 API Keys (Required)", open=True):
            openai_api = gr.Textbox(
                label="OpenAI API Key",
                placeholder="sk-...",
                type="password"
            )
            hf_api = gr.Textbox(
                label="HuggingFace API Token",
                placeholder="hf_...",
                type="password"
            )

        with gr.Accordion("📂 Upload Files for RAG", open=False):
            rag_files = gr.File(
                label="Upload documents (PDF, TXT)",
                type="filepath",             # returns file paths
                file_types=[".pdf", ".txt"], # allowed types
                file_count="multiple"        # allow multiple uploads
            )


        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    bubble_full_width=False,
                    avatar_images=("https://cdn-icons-png.flaticon.com/512/147/147144.png", 
                                 "https://cdn-icons-png.flaticon.com/512/4712/4712027.png")
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Ask me anything! Example: 'Generate an image of a sunset'",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Send 🚀", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary", size="sm")
                    export_btn = gr.Button("Export 📁", variant="secondary", size="sm")
                    
            with gr.Column(scale=3):
                stats_display = gr.Markdown(agent_interface.get_stats_display())
                status_display = gr.Textbox(label="Status", value="Ready to chat!", lines=2, interactive=False)

        # Example queries
        with gr.Accordion("💡 Example Queries", open=False):
            gr.Examples(
                examples=[
                    ["Generate an image of a futuristic cyberpunk city"],
                    ["Calculate the compound interest on $10,000 at 5% for 10 years"],
                    ["Search for the latest news about artificial intelligence"],
                    ["Write a Python function to find prime numbers"],
                    ["What is the capital of Australia?"],
                    ["Summarize the key benefits of renewable energy"],
                    ["Search for current weather in New York"]
                ],
                inputs=msg
            )

        # Handlers
        def handle_message(message, history, openai_key, hf_key, uploaded_files):
            # Pass API keys and file list to agent
            os.environ["OPENAI_API_KEY"] = openai_key or ""
            os.environ["HF_API_KEY"] = hf_key or ""
            return agent_interface.process_query(
                message=message, 
                history=history
            )

        submit_btn.click(
            fn=handle_message,
            inputs=[msg, chatbot, openai_api, hf_api, rag_files],
            outputs=[chatbot, msg, stats_display, status_display]
        )

        msg.submit(
            fn=handle_message,
            inputs=[msg, chatbot, openai_api, hf_api, rag_files],
            outputs=[chatbot, msg, stats_display, status_display]
        )

        clear_btn.click(
            fn=agent_interface.clear_conversation,
            outputs=[chatbot, stats_display, status_display]
        )

        export_btn.click(
            fn=agent_interface.export_conversation,
            inputs=[chatbot],
            outputs=[status_display]
        )

    return demo


def launch_app():
    """Launch the Gradio application"""
    print("🚀 Starting Multimodal AI Agent Interface...")
    print("🔧 Initializing agent components...")
    
    # Create and launch interface
    demo = create_interface()
    
    print("✅ Interface ready!")
    print("🌐 Launching web interface...")
    print("📝 Note: You'll need to set your API keys in the interface")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=True,             # Enable debug mode
        show_error=True,        # Show errors in interface
        favicon_path=None,      # Add custom favicon if needed
        ssl_verify=False        # For development
    )

if __name__ == "__main__":
    try:
        launch_app()
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Failed to start application: {str(e)}")
        print("Please check your configuration and try again.")