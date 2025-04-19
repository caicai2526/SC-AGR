import csv

input_file = r'F:\RRT_data\tcga-subtyping\TCGA-NSCLC R50\label.csv'  # 替换为您的原始CSV文件�?
output_file = r'F:\RRT_data\tcga-subtyping\TCGA-NSCLC R50\label_1.csv'  # 您希望生成的输出文件�?

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # 写入表头
    writer.writerow(['case_id', 'slide_id', 'label'])
    
    for row in reader:
        slide_id, label = row
        # 假设case_id为slide_id的前两部分，�?-'分隔
        case_id = '-'.join(slide_id.split('-')[:2])
        writer.writerow([case_id, slide_id, label])
