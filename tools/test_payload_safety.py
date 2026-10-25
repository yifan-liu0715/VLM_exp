import io,json,pathlib,struct,tempfile,unittest,zipfile,sys
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import payload_safety as s
class ContentSafety(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); self.root=pathlib.Path(self.tmp.name); self.secret=b'ghp_'+b'X'*36
 def tearDown(self):self.tmp.cleanup()
 def test_boundary_secret(self):
  with self.assertRaises(s.Held):s.scan_chunks(io.BytesIO(b'a'*(s.CHUNK-8)+self.secret),[self.secret])
 def test_nested_archive_secret(self):
  inner=io.BytesIO()
  with zipfile.ZipFile(inner,'w') as z:z.writestr('source.py',b'api_token='+self.secret)
  p=self.root/'outer.zip'
  with zipfile.ZipFile(p,'w') as z:z.writestr('inner.zip',inner.getvalue())
  with self.assertRaises(s.Held):s.scan_archive(p,[self.secret])
 def test_archive_path_escape(self):
  p=self.root/'bad.zip'
  with zipfile.ZipFile(p,'w') as z:z.writestr('../credentials',b'data')
  with self.assertRaises(s.Held):s.scan_archive(p,[])
 def test_safetensors_metadata_secret(self):
  p=self.root/'weights.safetensors';h=json.dumps({'__metadata__':{'token':self.secret.decode()},'w':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode();p.write_bytes(struct.pack('<Q',len(h))+h+b'0000')
  with self.assertRaises(s.Held):s.scan_safetensors(p,[self.secret])
 def test_valid_safetensors(self):
  p=self.root/'weights.safetensors';h=json.dumps({'w':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode();p.write_bytes(struct.pack('<Q',len(h))+h+b'0000');self.assertIn(b'F32',s.scan_safetensors(p,[]))
 def test_safetensors_bad_range(self):
  p=self.root/'weights.safetensors';h=json.dumps({'w':{'dtype':'F32','shape':[1],'data_offsets':[0,40]}}).encode();p.write_bytes(struct.pack('<Q',len(h))+h+b'0000')
  with self.assertRaises(s.Held):s.scan_safetensors(p,[])
 def test_credential_filename_in_archive(self):
  p=self.root/'bad.zip'
  with zipfile.ZipFile(p,'w') as z:z.writestr('folder/.credentials.json',b'{}')
  with self.assertRaises(s.Held):s.scan_archive(p,[])
 def test_source_change(self):
  p=self.root/'code.py';p.write_bytes(b'x=1');st=p.stat();row={'path':'code.py','size':st.st_size,'mtime_ns':st.st_mtime_ns};p.write_bytes(b'x=12')
  with self.assertRaises(s.Held):s.check_source(row,self.root)
 def test_archive_clean_roundtrip(self):
  p=self.root/'clean.zip'
  with zipfile.ZipFile(p,'w') as z:z.writestr('source.py',b'print(42)')
  s.scan_archive(p,[])
if __name__=='__main__':unittest.main()
