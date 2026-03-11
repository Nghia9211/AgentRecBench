import json
import os

def split_json_by_source(input_file, prefix_type):
    
    if not os.path.exists(input_file):
        print(f"❌ Not found: {input_file}")
        return

    print(f"🔄 Processing File: {input_file} ...")

    file_handles = {}
    
    count = 0
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                    source = data.get('source')
                    
                    if source:
                        if source not in file_handles:
                            output_filename = f"{prefix_type}_{source}.json"

                            file_handles[source] = open(output_filename, 'w', encoding='utf-8')
                        
                        file_handles[source].write(line + '\n')
                        count += 1
                        
                        if count % 10000 == 0:
                            print(f"  Processed {count} lines...", end='\r')
                            
                except json.JSONDecodeError:
                    continue 

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    
    finally:
        for source, f_handle in file_handles.items():
            f_handle.close()
            
    print(f"\n✅ Successfully {input_file}! Total {count} splited line.")
    print("-" * 30)

split_json_by_source('../../dataset/output_data_all/review.json', 'review')
split_json_by_source('../../dataset/output_data_all/item.json', 'item')
split_json_by_source('../../dataset/output_data_all/user.json', 'user')