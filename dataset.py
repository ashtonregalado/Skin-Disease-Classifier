import os

base_path = "SkinDisease/Train"

classes = os.listdir(base_path)
print("Classes:", classes)
print("Total classes:", len(classes))