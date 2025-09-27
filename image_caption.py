import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import logging

class ImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            print("Image captioning model loaded")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_caption(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(**inputs, max_length=50, num_beams=5)
            
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            logging.error(f"Error generating caption: {e}")
            return None

def get_image_captioner():
    return ImageCaptioner()