import json
import os
import sys
from datetime import datetime

class ReviewProcessor:
    """
    Lớp xử lý và chia tách dữ liệu đánh giá (Review).
    
    Cải tiến:
    1. Hỗ trợ đọc cả JSON Array ([...]) và JSON Lines.
    2. Chuẩn hóa thời gian (Timestamp) cho Amazon, Yelp, Goodreads.
    3. Chia tách theo logic: Review mới nhất là Short-term, Lịch sử cũ là Long-term.
    """

    def __init__(self, target_source='yelp'):
        """
        Khởi tạo processor.
        :param target_source: 'yelp', 'amazon', hoặc 'goodreads' để định hình cách parse ngày tháng.
        """
        self.target_source = target_source
        self.all_reviews = []
        self.sorted_reviews = []
        self.short_term_context = []
        self.long_term_context = []
        print(f"ReviewProcessor đã được khởi tạo cho nguồn: {self.target_source}")

    @staticmethod
    def get_normalized_timestamp(data, source):
        """
        Hàm tĩnh chuẩn hóa thời gian về dạng Unix Timestamp (milliseconds)
        để thống nhất cách xử lý cho mọi dataset.
        """
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
            # print(f"Lỗi parse ngày: {e}")
            return 0

    def load_reviews(self, input_file):
        self.all_reviews = input_file
            

    def process_and_split(self):
        """
        Thực hiện xử lý timestamp, sắp xếp và chia tách.
        Logic: 
        1. Chuẩn hóa timestamp.
        2. Sort tăng dần theo thời gian.
        3. Short-term = Item cuối cùng (mới nhất).
        4. Long-term = Tất cả items trước đó.
        """
        if not self.all_reviews:
            print("Cảnh báo: Không có dữ liệu để xử lý.")
            return

        # 1. Chuẩn hóa timestamp cho từng review
        valid_reviews = []
        for review in self.all_reviews:
            # Ưu tiên lấy source từ data, nếu không có thì dùng default cấu hình
            source = review.get('source', self.target_source)
            
            ts = self.get_normalized_timestamp(review, source)
            review['timestamp_norm'] = ts # Lưu field tạm để sort
            
            # Chỉ xử lý các review có timestamp hợp lệ (> 0)
            if ts > 0:
                valid_reviews.append(review)
            else:
                review['timestamp_norm'] = 0
                valid_reviews.append(review)

        self.sorted_reviews = sorted(valid_reviews, key=lambda x: x['timestamp_norm'])

        if not self.sorted_reviews:
            print("Không có review nào hợp lệ sau khi xử lý.")
            return

        if len(self.sorted_reviews) == 1:
            print("Chỉ có 1 review, không thể chia Long-term/Short-term. Mặc định vào Short-term.")
            self.short_term_context = [self.sorted_reviews[0]]
            self.long_term_context = []
        else:
            self.short_term_context = [self.sorted_reviews[-1]]   
            self.long_term_context = self.sorted_reviews[:-1]    

        for rv in self.sorted_reviews:
            rv.pop('timestamp_norm', None)

        print("\n--- KẾT QUẢ XỬ LÝ ---")
        print(f"Tổng số review: {len(self.sorted_reviews)}")
        print(f"Short Term Context (Mới nhất): {len(self.short_term_context)} review")
        print(f"Long Term Context (Lịch sử): {len(self.long_term_context)} review")
        
        # In thử ngày tháng để kiểm tra
        if self.short_term_context:
            last_date = self.short_term_context[0].get('date') or self.short_term_context[0].get('timestamp')
            print(f"Review mới nhất vào lúc: {last_date}")

    @staticmethod
    def save_to_json(data, filename):
        """Ghi dữ liệu ra file JSON"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Ghi dạng Array đẹp để dễ nhìn
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"-> Đã ghi file: {filename}")
        except Exception as e:
            print(f"Lỗi ghi file {filename}: {e}")

# --- KHỐI CHẠY CHÍNH ---
if __name__ == "__main__":
    # Cấu hình tên file (Bạn sửa tên file input ở đây)
    # File này chứa list các review của user (dạng Array [...])
    INPUT_FILENAME = "review.json"  # Đặt tên file input của bạn ở đây
    
    OUTPUT_LONG_TERM = "long_term_context.json"
    OUTPUT_SHORT_TERM = "short_term_context.json"
    
    # 1. Khởi tạo (Chọn source là 'yelp' vì dữ liệu bạn đưa là yelp)
    processor = ReviewProcessor(target_source='yelp')
    
    # 2. Load dữ liệu
    if processor.load_reviews(INPUT_FILENAME):
        
        # 3. Xử lý và tách (không cần nhập i, k, m nữa)
        processor.process_and_split()

        short_term = processor.short_term_context
        long_term = processor.long_term_context
        
        # 4. Lưu kết quả
        ReviewProcessor.save_to_json(processor.short_term_context, OUTPUT_SHORT_TERM)
        ReviewProcessor.save_to_json(processor.long_term_context, OUTPUT_LONG_TERM)
        
        print("\nHoàn tất quá trình tách file.")