import torch

# 请确认这里的文件路径正确
ckpt_path = 'pretrained_models/replay_buffer_0.pt'

print(f"正在加载文件: {ckpt_path} ...")
try:
    data = torch.load(ckpt_path, map_location='cpu')
    print(f"数据总类型: {type(data)}")

    if isinstance(data, list):
        print(f"这是一个列表 (List)，长度: {len(data)}")
        if len(data) > 0:
            print(f"列表第 0 项的类型: {type(data[0])}")
            # 如果第0项是 tensor，打印形状
            if torch.is_tensor(data[0]):
                print(f"列表第 0 项 Tensor Shape: {data[0].shape}")

    elif isinstance(data, dict):
        print(f"这是一个字典 (Dict)，包含的 Keys: {list(data.keys())[:5]}")

    else:
        print("既不是 List 也不是 Dict，请检查文件内容。")

except Exception as e:
    print(f"加载出错: {e}")