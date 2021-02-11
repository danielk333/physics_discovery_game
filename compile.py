import sys
import compileall
import pathlib
import shutil
import os

if __name__=='__main__':
    target = pathlib.Path(sys.argv[1])
    target.mkdir(exist_ok=True)

    target_lib = target / 'game'

    source = pathlib.Path('./game')
    
    shutil.copytree(source, target_lib)
    compileall.compile_dir(target_lib, force=True)

    files = target_lib.glob('*.py')

    for file in files:
        print(f'Removing {file}')
        os.remove(file)

    files = (target_lib / '__pycache__' ).glob('*.pyc')

    for file in files:
        print(f'Moving {file}')
        shutil.move(file, pathlib.Path(file).parents[1] / f'{pathlib.Path(file).name.split(".")[0]}.pyc')

    shutil.copy(pathlib.Path('./run.py'), target / 'run.py')