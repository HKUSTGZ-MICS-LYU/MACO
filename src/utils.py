import os

def check_suffix_file(dir: str, suffix: str) -> bool:
  # check if it's an empty folder
  if not any(os.scandir(dir)):
    return False
  for entry in os.scandir(dir):
    if entry.is_dir():
      folder_path = entry.path
      stats_files = [file for file in os.listdir(folder_path) if file.endswith(suffix)]
      if not stats_files:
        return False

  return True