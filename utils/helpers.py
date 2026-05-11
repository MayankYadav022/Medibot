import os, json
def ensure_dirs(*paths):
    for p in paths:
        if p:  # avoid None / empty paths
            os.makedirs(p, exist_ok=True)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # 🔥 ensure dir exists
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    if not os.path.exists(path):  # 🔥 safety check
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_txt(path):
    if not os.path.exists(path):  # 🔥 safety check
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
def write_txt(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # 🔥 ensure dir exists
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)