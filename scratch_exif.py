import os, glob
from PIL import Image, ExifTags

for l in ["H", "G", "P", "J"]:
    folder = os.path.join("data", "AzSLD_Fingerspelling", l)
    for f in sorted(glob.glob(os.path.join(folder, "*.*")))[:5]:
        im = Image.open(f)
        exif = im._getexif()
        exif_orient = None
        if exif:
            for tag, value in exif.items():
                if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == "Orientation":
                    exif_orient = value
        print(f"{l} ({os.path.basename(f)}): size={im.size}, EXIF Orientation={exif_orient}")

