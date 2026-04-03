from pathlib import Path
import shutil

def prepare(raw: Path, public: Path, private: Path) -> None:
    '''Copy pre-split public/private directories from raw data.
    TODO: Verify the folder names match your dataset structure.
    '''
    raw_public = raw / 'public'
    raw_private = raw / 'private'

    if not raw_public.exists():
        raise FileNotFoundError(f"Expected 'public' directory in raw data at {raw_public}")
    if not raw_private.exists():
        raise FileNotFoundError(f"Expected 'private' directory in raw data at {raw_private}")

    for item in raw_public.iterdir():
        dest = public / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    for item in raw_private.iterdir():
        dest = private / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)