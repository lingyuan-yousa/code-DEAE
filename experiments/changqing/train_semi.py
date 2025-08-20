import runpy
from pathlib import Path
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    script_dir = project_root / 'DEAE-torch-changqing'
    script_path = script_dir / 'semi-supervised-deae.py'
    sys.path.insert(0, str(script_dir))
    runpy.run_path(str(script_path), run_name='__main__')


if __name__ == '__main__':
    main()


