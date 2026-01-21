import shutil
import sys


from pathlib import Path


def main():
    base = Path(__file__).parent.parent
    stub_path = base / 'tests' / 'models' / 'beat_stub.onnx'
    reference_path = base / 'tests' / 'models' / 'beat_reference.onnx'
    try:
        shutil.copyfile(stub_path, reference_path)
        print(f'Copied {stub_path} -> {reference_path}')
    except (FileNotFoundError, OSError) as e:
        print(f'Error copying {stub_path} to {reference_path}: {e}')
        sys.exit(1)

if __name__ == "__main__":
    main()
