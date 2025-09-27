import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import logging

class TrainedToxicClassifier:
    def __init__(self, model_path='toxic_classifier_trained'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Trained model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def classify(self, text):
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, 
                padding=True, max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)
            
            label = "toxic" if predicted_class.item() == 1 else "non-toxic"
            toxicity_score = confidence.item() if label == "toxic" else 1 - confidence.item()
            
            return label, toxicity_score
            
        except Exception as e:
            logging.error(f"Classification error: {e}")
            return "non-toxic", 0.1

def get_classifier():
    return TrainedToxicClassifier()