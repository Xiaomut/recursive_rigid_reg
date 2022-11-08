import re
import sys

sys.path.append("../")
sys.path.append("./")
from utils.base_util import loadJson, saveJson

log_file = "files/coords.log"

with open(log_file, 'r') as f:
    r = f.read()

coords = re.findall("\[.*?\]", r)
coords = [eval(c.replace(' ', ', ')) for c in coords]
print(coords)

jsonfile = "files/train_coordinate.json"
images_dir = loadJson(jsonfile)

for (k, v), i in zip(images_dir.items(), range(0, len(coords), 2)):
    v["imgA"] = coords[i]
    v["imgB"] = coords[i + 1]

print(images_dir)
saveJson(images_dir, jsonfile)