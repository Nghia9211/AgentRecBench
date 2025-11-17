import json
from datetime import datetime
import sys

class ReviewProcessor:
    """
    Một lớp để xử lý và chia tách dữ liệu đánh giá của người dùng.

    Lớp này đóng gói các logic để:
    1. Tải dữ liệu đánh giá từ một tệp JSON.
    2. Sắp xếp các đánh giá theo ngày cập nhật (mới nhất trước tiên).
    3. Chia tách dữ liệu thành "bối cảnh ngắn hạn" (short-term context) và 
       "bối cảnh dài hạn" (long-term context) dựa trên các tham số có thể cấu hình.
    4. Cung cấp các phương thức để lưu kết quả ra tệp JSON.
    """
    
    # Định dạng ngày tháng của Goodreads, sử dụng như một hằng số của lớp
    DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
    SECONDS_IN_A_DAY = 24 * 60 * 60

    def __init__(self):
        """Khởi tạo một thực thể ReviewProcessor."""
        self.all_reviews = []
        self.sorted_reviews = []
        self.short_term_context = []
        self.long_term_context = []
        print("ReviewProcessor đã được khởi tạo.")

    @staticmethod
    def _get_sort_key(review):
        """
        Phương thức tĩnh (static method) để lấy khóa sắp xếp từ một đánh giá.

        Nó chuyển đổi chuỗi ngày tháng ("date_updated") thành một timestamp số nguyên 
        và thêm trường 'date_updated_int' vào dictionary của review.
        
        Tham số:
            review (dict): Một đối tượng đánh giá.
            
        Trả về:
            int: Timestamp của ngày cập nhật, hoặc 0 nếu có lỗi.
        """
        date_str = review.get("date_updated")
        timestamp_int = 0

        if not date_str:
            print(f"Cảnh báo: Đánh giá không có 'date_updated'. ID: {review.get('review_id', 'N/A')}")
        else:
            try:
                dt_object = datetime.strptime(date_str, ReviewProcessor.DATE_FORMAT)
                timestamp_int = int(dt_object.timestamp())
            except ValueError as e:
                print(f"Lỗi phân tích ngày '{date_str}' (ID: {review.get('review_id', 'N/A')}): {e}.")
        
        review['date_updated_int'] = timestamp_int
        return timestamp_int

    def load_reviews(self, input_file):
        """
        Tải dữ liệu đánh giá từ một tệp JSON.

        Tham số:
            input_filename (str): Đường dẫn đến tệp JSON đầu vào.

        Trả về:
            bool: True nếu tải thành công, False nếu có lỗi.
        """

        self.all_reviews = input_file
        return False

    def process_and_split(self, days_window_i, max_items_k, max_items_m):
        """
        Thực hiện sắp xếp và chia tách các đánh giá đã được tải.

        Tham số:
            days_window_i (int): Số ngày tối đa cho bối cảnh ngắn hạn.
            max_items_k (int): Số lượng item tối đa cho bối cảnh ngắn hạn.
            max_items_m (int): Số lượng item tối đa cho bối cảnh dài hạn.
        """
        if not self.all_reviews:
            print("Cảnh báo: Không có dữ liệu đánh giá để xử lý. Vui lòng gọi `load_reviews` trước.")
            return

        # 1. Sắp xếp tất cả các đánh giá
        self.sorted_reviews = sorted(self.all_reviews, key=self._get_sort_key, reverse=True)
        
        if not self.sorted_reviews:
            return

        # 2. Logic chia tách
        session_duration_seconds = days_window_i * self.SECONDS_IN_A_DAY
        latest_timestamp = self.sorted_reviews[0].get('date_updated_int', 0)
        time_threshold = latest_timestamp - session_duration_seconds

        time_window_reviews = []
        long_term_reviews_temp = []

        for review in self.sorted_reviews:
            review_timestamp = review.get('date_updated_int', 0)
            if review_timestamp > 0 and review_timestamp >= time_threshold:
                time_window_reviews.append(review)
            else:
                long_term_reviews_temp.append(review)
        
        # 3. Áp dụng giới hạn K và M
        self.short_term_context = time_window_reviews[:max_items_k]
        self.long_term_context = long_term_reviews_temp[:max_items_m]

        print("\n--- THÔNG TIN CHIA TÁCH ---")
        print(f"Timestamp Mới nhất: {datetime.fromtimestamp(latest_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Mốc thời gian (i={days_window_i} ngày trước): {datetime.fromtimestamp(time_threshold).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Giới hạn K (short-term): {max_items_k} mục.")
        print(f"Giới hạn M (long-term): {max_items_m} mục.")
        print(f"Số lượng thực tế trong 'Short Term Context': {len(self.short_term_context)} mục.")
        print(f"Số lượng thực tế trong 'Long Term Context': {len(self.long_term_context)} mục.")

    @staticmethod
    def save_to_json(data, filename):
        """
        Phương thức tĩnh để ghi dữ liệu vào một tệp JSON.
        
        Tham số:
            data (list): Dữ liệu (danh sách các dictionary) cần ghi.
            filename (str): Tên tệp đầu ra.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"THÀNH CÔNG: Đã ghi {len(data)} mục vào tệp: {filename}")
        except Exception as e:
            print(f"LỖI khi ghi tệp JSON '{filename}': {e}")

# --- KHỐI MÃ ĐỂ CHẠY VÀ KIỂM THỬ TRỰC TIẾP TỆP NÀY ---
# Khối này sẽ chỉ thực thi khi bạn chạy `python review_processor.py`
if __name__ == "__main__":
    # Các hằng số cho việc chạy thử nghiệm
    INPUT_FILENAME = r"user_4cbecbc15af3db041a8e0f594c642bb5_history_review.json" 
    OUTPUT_FILENAME = "sorted_reviews.json"
    SHORT_TERM_FILENAME = "short_term_context.json"
    LONG_TERM_FILENAME = "long_term_context.json"
    
    # 1. Khởi tạo đối tượng
    processor = ReviewProcessor()
    
    # 2. Tải dữ liệu
    if not processor.load_reviews(INPUT_FILENAME):
        sys.exit(1) # Thoát nếu không tải được tệp

    # 3. Lấy đầu vào từ người dùng
    try:
        days_i = int(input("Nhập i (số ngày tối đa cho Short Term Context): "))
        items_k = int(input("Nhập k (số item tối đa cho Short Term Context): "))
        items_m = int(input("Nhập m (số item tối đa cho Long Term Context): "))
        if days_i <= 0 or items_k <= 0 or items_m <= 0:
            print("Lỗi: Các giá trị i, k, m phải là số nguyên dương.")
            sys.exit(1)
    except ValueError:
        print("Lỗi: Đầu vào phải là số nguyên.")
        sys.exit(1)
    
    # 4. Xử lý và chia tách
    processor.process_and_split(days_i, items_k, items_m)
    
    # 5. Lưu kết quả
    # Sử dụng phương thức tĩnh của lớp để lưu
    ReviewProcessor.save_to_json(processor.short_term_context, SHORT_TERM_FILENAME)
    ReviewProcessor.save_to_json(processor.long_term_context, LONG_TERM_FILENAME)
    ReviewProcessor.save_to_json(processor.sorted_reviews, OUTPUT_FILENAME)
    
    print("\nChương trình đã hoàn thành.")