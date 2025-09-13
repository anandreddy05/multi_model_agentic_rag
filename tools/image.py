import requests
import base64
from io import BytesIO
from langchain.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

hf_api_key = os.getenv("HF_API_KEY")
if hf_api_key is not None:
    os.environ["HF_API_KEY"] = hf_api_key
    
@tool
def generate_image_tool(prompt: str, model_id: str = "black-forest-labs/FLUX.1-dev") -> str:
    """Generate an image from text prompt and save it locally. Returns the file path."""
    print(f"[Tool: generate_image] Invoked with prompt: {prompt}")
    
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            os.makedirs("generated_images", exist_ok=True)
            
            # Generate filename from prompt (sanitized)
            safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = f"generated_images/{safe_prompt.replace(' ', '_')}.png"
            
            # Save image
            with open(filename, "wb") as f:
                f.write(response.content)
            
            return f"Image generated successfully and saved as: {filename}"
        else:
            return f"Failed to generate image. Error: {response.status_code} - {response.text}"
    
    except Exception as e:
        return f"Error generating image: {str(e)}"