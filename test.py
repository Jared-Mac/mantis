import sys
import torchdistill
import os

print("torchdistill module loaded from:")
print(torchdistill.__file__) # This tells you where the *currently loaded* torchdistill __init__.py is

print("\nPaths in sys.path where 'torchdistill' might be found:")
for p in sys.path:
    potential_path = os.path.join(p, "torchdistill")
    if os.path.exists(potential_path):
        print(f"  Found: {potential_path} (Type: {'Directory' if os.path.isdir(potential_path) else 'File'})")
    
    # Check for .egg-link files if you ever used `pip install -e`
    potential_egg_link = os.path.join(p, "torchdistill.egg-link")
    if os.path.exists(potential_egg_link):
        print(f"  Found egg-link: {potential_egg_link}")
        with open(potential_egg_link, 'r') as f:
            print(f"    -> points to: {f.read().strip()}")