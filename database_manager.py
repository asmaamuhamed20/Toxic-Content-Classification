import pandas as pd
import os
from datetime import datetime
import csv

class DatabaseManager:
    def __init__(self, csv_file="toxic_content_database.csv"):
        self.csv_file = csv_file
        self._initialize_database()
    
    def _initialize_database(self):
        if not os.path.exists(self.csv_file):
            df = pd.DataFrame(columns=[
                'timestamp', 'input_type', 'input_content', 
                'generated_caption', 'classification_result', 'toxicity_score'
            ])
            df.to_csv(self.csv_file, index=False)
    
    def add_entry(self, input_type, input_content, generated_caption, classification_result, toxicity_score):
        new_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input_type': input_type,
            'input_content': str(input_content)[:500],
            'generated_caption': str(generated_caption)[:500],
            'classification_result': classification_result,
            'toxicity_score': toxicity_score
        }
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=new_entry.keys())
            writer.writerow(new_entry)
    
    def get_all_entries(self):
        try:
            return pd.read_csv(self.csv_file)
        except:
            return pd.DataFrame()

    def get_statistics(self):
        df = self.get_all_entries()
        if df.empty:
            return {"total_entries": 0}
        
        return {
            "total_entries": len(df),
            "text_inputs": len(df[df['input_type'] == 'text']),
            "image_inputs": len(df[df['input_type'] == 'image']),
            "toxic_content": len(df[df['classification_result'] == 'toxic'])
        }