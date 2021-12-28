import os
import shutil
import pathlib

def remove_files(mydir):

    mydir=os.getcwd()
    filelist = [ f for f in os.listdir(mydir) if f.endswith(".csv") ]
    for f in filelist:
        os.remove(os.path.join(mydir, f))

def move_files(source,destination):

    dest=pathlib.Path(destination)
    if dest.exists():
        pass
    else:
        os.mkdir(dest)
    filelist = [f for f in os.listdir(source) if f.endswith(".csv")]
    for f in filelist:
        shutil.move(os.path.join(source,f), os.path.join(destination,f))

