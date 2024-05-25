import pandas as pd


def clean_excel(input_file, output_file):
    # 读取 Excel 文件
    df = pd.read_excel(input_file)

    # 删除 y 列等于 0 的行
    df = df[df['Y'] != 0]

    # 保存处理后的数据到新的 Excel 文件
    df.to_excel(output_file, index=False)

    print(f"处理完成，已保存为 {output_file}")


# 使用示例
input_excel_file = '../data/train data.xlsx'  # 输入的 Excel 文件名
output_excel_file = 'output.xlsx'  # 输出的 Excel 文件名

# 调用函数进行处理和保存
clean_excel(input_excel_file, output_excel_file)
