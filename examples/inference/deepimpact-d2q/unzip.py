import zipfile

with zipfile.ZipFile("deepimpact-bert-base.zip", 'r') as zip_ref:
    zip_ref.extractall("./")