import runpy
import sys
from pathlib import Path

examples_dir = Path(__file__).parent

if len(sys.argv) < 2:
    print('Available examples:')
    for f in sorted(examples_dir.glob('*.py')):
        if f.name != '__main__.py':
            print(f'  {f.stem}')
    sys.exit(0)

name = sys.argv[1]
script = examples_dir / f'{name}.py'
if not script.exists():
    print(f'Example not found: {name}')
    sys.exit(1)

runpy.run_path(str(script), run_name='__main__')
