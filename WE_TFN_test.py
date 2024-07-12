import h5py

# 打开 .h5 文件
file_path = 'E:\PythonProject\WE-TFN\data\BJ16_M32x32_T30_InOut.h5'  # 替换为你的 .h5 文件路径
with h5py.File(file_path, 'r') as f:
    def print_attrs(name, obj):
        print(f"{name}:")
        for key, value in obj.attrs.items():
            print(f"  {key}: {value}")


    # 列出所有数据集
    print("文件结构:")
    f.visititems(print_attrs)

    # 查看具体数据集的信息
    for dataset_name in f.keys():
        dataset = f[dataset_name]
        print(f"\n数据集 '{dataset_name}' 信息:")
        print(f"  形状: {dataset.shape}")
        print(f"  数据类型: {dataset.dtype}")
        if 'columns' in dataset.attrs:
            print(f"  列名: {dataset.attrs['columns']}")

# 示例输出数据集内容
with h5py.File(file_path, 'r') as f:
    for dataset_name in f.keys():
        dataset = f[dataset_name]
        print(f"\n数据集 '{dataset_name}' 的前5条记录:")
        print(dataset[:5])
