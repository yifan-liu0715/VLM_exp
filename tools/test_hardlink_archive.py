"""Regression: distinct source paths sharing an inode must restore as regular files."""
import json,os,pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import upload_archive_remote as archive
import restore_release as restore
class HardlinkArchive(unittest.TestCase):
 def test_hardlinks_preserve_both_source_paths(self):
  with tempfile.TemporaryDirectory() as d:
   home=pathlib.Path(d);root=home/'source';root.mkdir();out=home/'export';out.mkdir();dest=home/'restored';dest.mkdir()
   a=root/'a.mp4';b=root/'b.mp4';a.write_bytes(b'original-source-data'*4096);os.link(a,b)
   rows=[]
   for p in (a,b):
    st=p.stat();rows.append({'path':p.name,'size':st.st_size,'mtime_ns':st.st_mtime_ns,'mode':st.st_mode})
   job={'id':'hardlinks','kind':'tar_zstd','asset_name':'hardlinks.tar.zst','files':rows,'logical_bytes':sum(r['size'] for r in rows),'needs_security_review':False}
   p,receipt=archive.produce(job,root,out)
   self.assertEqual(restore.extract_tar(p,dest,job),2)
   self.assertEqual((dest/'workspace/a.mp4').read_bytes(),a.read_bytes())
   self.assertEqual((dest/'workspace/b.mp4').read_bytes(),a.read_bytes())
   self.assertEqual(a.stat().st_ino,b.stat().st_ino)
if __name__=='__main__':unittest.main()
