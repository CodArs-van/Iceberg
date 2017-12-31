import os
import shutil
import stat
import threading

def copytree(src, dst, symlinks = False, ignore = None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    
    lst = os.listdir(src)
    if ignore:
        excludes = ignore(src, lst)
        lst = [item for item in lst if item not in excludes]
    
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                # lchmod not available
                pass
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
            
def backup(sdir, ddir, symlinks, ignore, nsec=120.0):
    threading.Timer(nsec, backup, [sdir, ddir, symlinks, ignore, nsec]).start()
    print('execute backup operation')
    copytree(sdir, ddir, symlinks, ignore)