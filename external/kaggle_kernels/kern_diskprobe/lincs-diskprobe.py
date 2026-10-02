# Disk-capacity probe for P7's staging (review 038, "could not assess"): free CPU session, same image. No data read.
import os, shutil, subprocess
print(subprocess.run(['df', '-h'], capture_output=True, text=True).stdout, flush=True)
for p in ('/tmp', '/kaggle/working', '/kaggle/temp' if os.path.isdir('/kaggle/temp') else '/tmp'):
    u = shutil.disk_usage(p)
    print('%-16s total %.1f GB free %.1f GB dev %d' % (p, u.total / 1e9, u.free / 1e9, os.stat(p).st_dev), flush=True)
