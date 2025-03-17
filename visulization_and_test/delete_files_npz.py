import os
import shutil


def delete_files_and_empty_folders(txt_file):
    """
    根据 txt 文件中的路径列表，删除文件及其所在的空文件夹。
    Args:
        txt_file (str): 包含要删除的文件路径的 txt 文件。
    """
    if not os.path.exists(txt_file):
        print(f"Error: The file '{txt_file}' does not exist.")
        return

    # 读取 txt 文件中的路径
    with open(txt_file, "r") as f:
        file_paths = f.read().splitlines()

    # 初始化统计信息
    total_files = len(file_paths)
    deleted_files = 0
    missing_files = 0
    deleted_folders = 0

    # 删除文件并检查空文件夹
    for file_path in file_paths:
        # 删除文件
        if os.path.exists(file_path):
            try:
                os.remove(file_path)  # 删除文件
                print(f"Deleted file: {file_path}")
                deleted_files += 1
            except Exception as e:
                print(f"Error deleting file '{file_path}': {e}")
        else:
            print(f"File not found: {file_path}")
            missing_files += 1

        # 检查并删除空的类别文件夹
        folder_path = os.path.dirname(file_path)
        if os.path.exists(folder_path) and len(os.listdir(folder_path)) == 0:
            try:
                shutil.rmtree(folder_path)  # 删除空文件夹
                print(f"Deleted empty folder: {folder_path}")
                deleted_folders += 1
            except Exception as e:
                print(f"Error deleting folder '{folder_path}': {e}")

    # 检查整个目录树中是否还有空文件夹
    base_dir = os.path.dirname(os.path.commonprefix(file_paths))  # 获取最上层的公共父目录
    for root, dirs, _ in os.walk(base_dir, topdown=False):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            if len(os.listdir(folder_path)) == 0:  # 如果文件夹为空
                try:
                    shutil.rmtree(folder_path)
                    print(f"Deleted empty folder: {folder_path}")
                    deleted_folders += 1
                except Exception as e:
                    print(f"Error deleting folder '{folder_path}': {e}")

    # 打印总结信息
    print("\nSummary:")
    print(f"Total files listed: {total_files}")
    print(f"Deleted files: {deleted_files}")
    print(f"Missing files: {missing_files}")
    print(f"Deleted empty folders: {deleted_folders}")


if __name__ == "__main__":
    # 输入 txt 文件路径
    txt_file = "insufficient_points.txt"  # 修改为实际的 txt 文件路径

    # 执行批量删除操作
    delete_files_and_empty_folders(txt_file)


