import torch
print(torch.cuda.is_available())  # 应返回 True
print(torch.cuda.device_count())  # 应返回可用的 GPU 数量
print(torch.cuda.get_device_name(0))  # 应显示 GPU 名称
