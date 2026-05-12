import numpy as np

# 加载 .npz 文件
data = np.load("/root/wish/oneformer3d_main/predictions_clean/pred_0.npz")

# 查看文件包含的数组名称
print("文件中的数组名称:", data.files)

# 提取特定数组（例如 'masks'）
masks = data["masks"]  # 假设文件里有一个叫 'masks' 的数组
print("masks 的形状:", masks.shape)

# 关闭文件（可选，但推荐）
data.close()