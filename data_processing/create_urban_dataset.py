import os
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_PATH = os.path.join(BASE_DIR, "dataset")
NEW_PATH = os.path.join(BASE_DIR, "dataset_urban")

urban_classes = {
    1: 0,   # bicycle
    5: 1,   # bus
    6: 2,   # car
    13: 3,  # motorbike
    14: 4,  # person
    18: 5   # train
}

splits = ["train", "val"]

for split in splits:
    img_src = os.path.join(BASE_PATH, "images", split)
    lbl_src = os.path.join(BASE_PATH, "labels", split)

    img_dst = os.path.join(NEW_PATH, "images", split)
    lbl_dst = os.path.join(NEW_PATH, "labels", split)

    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    for file in os.listdir(lbl_src):
        label_path = os.path.join(lbl_src, file)

        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])

            if class_id in urban_classes:
                new_id = urban_classes[class_id]
                parts[0] = str(new_id)
                new_lines.append(" ".join(parts))

     
        if len(new_lines) > 0:
          
            img_name = file.replace(".txt", ".jpg")
            shutil.copy(os.path.join(img_src, img_name), img_dst)

         
            with open(os.path.join(lbl_dst, file), "w") as f:
                f.write("\n".join(new_lines))

print("Urban dataset created successfully.")
