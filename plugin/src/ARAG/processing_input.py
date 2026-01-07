import json
import os
import sys
from datetime import datetime

class ReviewProcessor:
    def __init__(self, target_source='yelp'):
        self.target_source = target_source
        self.all_reviews = []
        self.sorted_reviews = []
        self.short_term_context = []
        self.long_term_context = []
        print(f"ReviewProcessor đã được khởi tạo cho nguồn: {self.target_source}")

    @staticmethod
    def get_normalized_timestamp(data, source):
        try:
            if source == 'amazon':
                return int(data.get('timestamp', 0))
                
            elif source == 'yelp':
                date_str = data.get('date')
                if date_str:
                    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    return int(dt.timestamp() * 1000)
                    
            elif source == 'goodreads':
                date_str = data.get('date_added') or data.get('date_updated')
                if date_str:
                    try:
                        dt = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
                        return int(dt.timestamp() * 1000)
                    except ValueError:
                        pass
            return 0
        except Exception as e:
            return 0

    def load_reviews(self, input_file):
        self.all_reviews = input_file
        


    def process_and_split(self):
        if not self.all_reviews:
            return

        valid_reviews = []
        for review in self.all_reviews:
            source = review.get('source', self.target_source)
            
            ts = self.get_normalized_timestamp(review, source)
            review['timestamp_norm'] = ts
            
            if ts > 0:
                valid_reviews.append(review)
            else:
                review['timestamp_norm'] = 0
                valid_reviews.append(review)

        self.sorted_reviews = sorted(valid_reviews, key=lambda x: x['timestamp_norm'])

        if not self.sorted_reviews:
            return

        if len(self.sorted_reviews) == 1:
            self.short_term_context = [self.sorted_reviews[0]]
            self.long_term_context = []
        else:
            self.short_term_context = [self.sorted_reviews[-1]]   
            self.long_term_context = self.sorted_reviews[:-1]    

        for rv in self.sorted_reviews:
            rv.pop('timestamp_norm', None)

        if self.short_term_context:
            last_date = self.short_term_context[0].get('date') or self.short_term_context[0].get('timestamp')

    @staticmethod
    def save_to_json(data, filename):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f" WRITE : {filename}")
        except Exception as e:
            print(f"ERROR : {filename}: {e}")
