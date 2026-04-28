import os

base_path = "SkinDisease/train"

classes = os.listdir(base_path)
print("Classes:", classes)
print("Total classes:", len(classes))