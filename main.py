import os
import time
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ImageHandler(FileSystemEventHandler):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            if file_path.endswith(".jpg") or file_path.endswith(".png"):
                image = Image.open(file_path)
                # 在这里对新添加的图片进行处理
                print("New image added:", file_path)
                # 例如，显示图片
                image.show()

def monitor_folder(folder_path):
    event_handler = ImageHandler(folder_path)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# 指定文件夹路径
folder_path = "C:/Users/root/Desktop/数字人结果/图片"

# 开始监测文件夹中的变化并读取新添加的图片
monitor_folder(folder_path)
