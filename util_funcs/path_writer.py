import json
import os

dataset_path_base = "../../data/datasets/Robofarmer/"
paths = {
        "P01" :  os.path.join(dataset_path_base, "P01"), 
        "P02" :  os.path.join(dataset_path_base, "P02"), 
        "P03" :  os.path.join(dataset_path_base, "P03"), 
        "video" : "../../data/video/", 
        "dataset" : dataset_path_base,
        "features" : "../../data/datasets/features/robofarmer/",
}

json_file = "../../data/paths.json"

# Write the paths to the JSON file
with open(json_file, "w") as file:
    json.dump(paths, file, indent=4)

print(f"Paths have been written to {json_file}")
