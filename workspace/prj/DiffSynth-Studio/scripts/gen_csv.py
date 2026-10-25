import csv
import os

# 原始数据路径（直接读取原始 CSV）
input_csv = '/nvmedata/workspace2/users/nina/dataset_video_train/metadata.csv'
# 输出到项目目录下的 data 文件夹
output_csv = './data/dataset_video_train/metadata_train.csv'

# 确保输出目录存在
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

with open(input_csv, 'r') as f_in, open(output_csv, 'w', newline='') as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.writer(f_out)
    # 官方标准列名
    writer.writerow(['video', 'prompt', 'reference_image'])
    
    count = 0
    for row in reader:
        sample_id = row['sample_id']
        bucket = row['bucket']
        # 路径是相对于 dataset_base_path 的相对路径
        # video_path = f\"{bucket}/output_video/{sample_id}.mp4\"
        # reference_path = f\"{bucket}/reference/{sample_id}.jpg\"
        caption = row['caption']
        
        writer.writerow([video_path, caption, reference_path])
        count += 1

print(f'✅ 生成标准CSV: {output_csv}')
print(f'共处理 {count} 条数据')
print('列名: video, prompt, reference_image')
