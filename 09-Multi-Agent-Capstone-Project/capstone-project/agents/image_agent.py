import requests
import io
import os
from PIL import Image
from config import HUGGING_FACE_TOKEN


class ImageAgent:
    def __init__(self):
        self.api_url = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
        self.headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}

    def _improve_prompt(self, user_query):
        """
        Prompt Engineering: Adds quality keywords to the user query 
        to ensure the generated image looks professional.
        """
        # Example: 'concert' -> 'concert, highly detailed, 8k, cinematic lighting, professional photography'
        enhanced_prompt = f"{user_query}, high resolution, 8k, cinematic lighting, photorealistic, masterpiece"
        return enhanced_prompt

    def generate_image(self, user_query, save_path="generated-images/generated_image.png"):
        """Sends the request to Hugging Face and saves the output."""
        print(f"[*] Generating image for: {user_query}...")

        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        payload = {
            "inputs": self._improve_prompt(user_query),
            "options": {"wait_for_model": True}
        }

        try:
            response = requests.post(
                self.api_url, headers=self.headers, json=payload)

            if response.status_code == 200:
                image_bytes = response.content
                image = Image.open(io.BytesIO(image_bytes))
                image.save(save_path)
                return f"Success! Image saved to {save_path}"
            else:
                return f"Image API Error: {response.text}"

        except Exception as e:
            return f"Image Agent Error: {str(e)}"
