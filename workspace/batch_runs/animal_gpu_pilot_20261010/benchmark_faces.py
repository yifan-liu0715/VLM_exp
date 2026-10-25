"""Isolated CPU/CUDA YuNet parity and timing pilot. Never approves samples.

Decode convention matches OpenCV FaceDetectorYN's published implementation:
https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/face_detect.cpp
No change to production detector, model weights, privacy, or training gates.
"""
import argparse,json,time,hashlib,os,sys,collections,subprocess
from pathlib import Path
import cv2,numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Frames:
    def __init__(self,path,width,height,decoder,ffmpeg):
        self.w=width;self.h=height;self.cap=None;self.process=None
        if decoder=='opencv': self.cap=cv2.VideoCapture(str(path))
        else:self.process=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-i',str(path),'-vf',f'fps=16,scale={width}:{height}','-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    def read(self):
        if self.cap is not None:return self.cap.read()
        data=self.process.stdout.read(self.w*self.h*3)
        if not data:
            assert self.process.wait(timeout=60)==0,self.process.stderr.read().decode(errors='replace')
            return False,None
        assert len(data)==self.w*self.h*3,'Partial raw frame'
        return True,np.frombuffer(data,dtype=np.uint8).reshape(self.h,self.w,3)
    def release(self):
        if self.cap is not None:self.cap.release()
        if self.process is not None:
            if self.process.poll() is None:self.process.terminate()
            self.process.wait(timeout=60);self.process.stdout.close();self.process.stderr.close()
def decode(outputs,width,height):
    found=[]
    for stride in (8,16,32):
        scores=np.sqrt(np.clip(outputs[f'cls_{stride}'].reshape(-1),0,1)*np.clip(outputs[f'obj_{stride}'].reshape(-1),0,1))
        ids=np.flatnonzero(scores>=.6)
        if not len(ids):continue
        bb=outputs[f'bbox_{stride}'].reshape(-1,4)[ids]
        cols=width//stride
        xy=(np.column_stack((ids%cols,ids//cols))+bb[:,:2])*stride
        wh=np.exp(bb[:,2:])*stride
        found.extend(np.column_stack((xy-wh/2,wh,scores[ids])).tolist())
    if len(found)>1:
        keep=cv2.dnn.NMSBoxes([[int(v) for v in a[:4]] for a in found],[a[4] for a in found],.6,.3,eta=1.,top_k=5000)
        found=[found[int(i)] for i in np.asarray(keep).reshape(-1)]
    return found

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['local_cpu','remote_gpu'],required=True);ap.add_argument('--root',type=Path,default=Path(__file__).parent);ap.add_argument('--no-profile',action='store_true');ap.add_argument('--decoder',choices=['opencv','ffmpeg'],default='opencv');ap.add_argument('--ffmpeg',default='ffmpeg');args=ap.parse_args()
    root=args.root;m=json.loads((root/'transfer_manifest.json').read_text());cv2.setNumThreads(2)
    report=dict(mode=args.mode,opencv=cv2.__version__,opencv_threads=cv2.getNumThreads(),batch_size=1,score_threshold=.6,nms_threshold=.3,accepted_for_training=False,privacy_reviewed=False,semantic_reviewed=False,files=[],model_original_sha256=sha(root/m['model']))
    detector=None;session=None;shape=None;warmup_s=0.;init_s=0.;profile=None
    if args.mode=='remote_gpu':
        # Torch loads the CUDA/cuDNN runtime already present in this environment.
        import torch,onnxruntime as ort,onnx
        model=onnx.load(str(root/m['model']))
        original_weights=[hashlib.sha256(t.SerializeToString()).hexdigest() for t in model.graph.initializer]
        dims=model.graph.input[0].type.tensor_type.shape.dim
        for i,name in [(2,'image_height'),(3,'image_width')]:
            dims[i].ClearField('dim_value');dims[i].dim_param=name
        for output in model.graph.output: output.type.tensor_type.ClearField('shape')
        dynamic=root/'yunet_same_weights_dynamic_input.onnx';onnx.save(model,str(dynamic))
        assert original_weights==[hashlib.sha256(t.SerializeToString()).hexdigest() for t in onnx.load(str(dynamic)).graph.initializer]
        opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;opt.enable_profiling=not args.no_profile;opt.profile_file_prefix=str(root/'reports/cuda_profile');opt.log_severity_level=3
        opts={'device_id':0,'gpu_mem_limit':str(1024**3),'cudnn_conv_algo_search':'DEFAULT','use_tf32':'0'}
        start=time.perf_counter();session=ort.InferenceSession(str(dynamic),sess_options=opt,providers=[('CUDAExecutionProvider',opts),'CPUExecutionProvider']);init_s=time.perf_counter()-start
        assert session.get_providers()[0]=='CUDAExecutionProvider',session.get_providers()
        report.update(onnxruntime=ort.__version__,cuda_build=torch.version.cuda,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),providers=session.get_providers(),gpu_name=torch.cuda.get_device_name(0),cuda_memory_arena_limit_bytes=1024**3,model_weights_unchanged=True,dynamic_model_sha256=sha(dynamic))
        names=[o.name for o in session.get_outputs()];input_name=session.get_inputs()[0].name
    for f in m['files']:
        if f['kind']!='candidate':continue
        path=Path(f['local_path']) if args.mode=='local_cpu' else root/f['remote_relative']
        assert sha(path)==f['sha256'],path
        cap=cv2.VideoCapture(str(path));assert cap.isOpened(),path
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));assert (width,height)==(1280,720)
        cap.release();cap=Frames(path,width,height,args.decoder,args.ffmpeg)
        if shape!=(width,height):
            if args.mode=='local_cpu':detector=cv2.FaceDetectorYN.create(str(root/m['model']),'',(width,height),.6,.3,5000)
            shape=(width,height);ok,first=cap.read();assert ok;cap.release();cap=Frames(path,width,height,args.decoder,args.ffmpeg)
            t=time.perf_counter()
            for _ in range(3):
                if session:
                    padded=cv2.copyMakeBorder(first,0,(-height)%32,0,(-width)%32,cv2.BORDER_CONSTANT,value=0)
                    session.run(None,{input_name:cv2.dnn.blobFromImage(padded)})
                else:detector.detect(first)
            warmup_s+=time.perf_counter()-t
        n=0;hits=[];inference_s=0.;started=time.perf_counter()
        while True:
            ok,im=cap.read()
            if not ok:break
            t=time.perf_counter()
            if session:
                padded=cv2.copyMakeBorder(im,0,(-height)%32,0,(-width)%32,cv2.BORDER_CONSTANT,value=0)
                out=dict(zip(names,session.run(None,{input_name:cv2.dnn.blobFromImage(padded)})))
                faces=decode(out,padded.shape[1],padded.shape[0])
            else:
                _,rows=detector.detect(im);faces=[] if rows is None else [[*r[:4],r[-1]] for r in rows]
            inference_s+=time.perf_counter()-t
            for face in faces:hits.append(dict(frame_index=n,bbox_xywh=[float(v) for v in face[:4]],score=float(face[4])))
            n+=1
        cap.release();wall=time.perf_counter()-started;assert n==f['expected_frames'],(path,n,f['expected_frames'])
        row=dict(source_id=f['source_id'],event_id=f['event_id'],frames=n,sha256=f['sha256'],width=width,height=height,all_frames_checked=True,detector_seconds=inference_s,decode_and_detect_seconds=wall,hits=hits)
        report['files'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='hits'}),flush=True)
    if session and not args.no_profile:
        profile=session.end_profiling();events=json.loads(Path(profile).read_text());providers=collections.Counter();conv_providers=collections.Counter()
        for e in events:
            a=e.get('args',{});provider=a.get('provider')
            if provider:
                providers[provider]+=1
                if 'Conv' in a.get('op_name',''):conv_providers[provider]+=1
        report.update(profile_path=profile,profile_provider_events=dict(providers),profile_conv_events=dict(conv_providers),cuda_compute_verified=bool(conv_providers['CUDAExecutionProvider']) and not conv_providers['CPUExecutionProvider'])
        assert report['cuda_compute_verified'],report['profile_conv_events']
    elif session:
        prior=json.loads((root/'reports/remote_gpu.json').read_text())
        assert prior['cuda_compute_verified'] and prior['dynamic_model_sha256']==report['dynamic_model_sha256'] and prior['cuda_visible_devices']==report['cuda_visible_devices']
        report.update(cuda_compute_verified_in_prior_profile=True,prior_profile_path=prior['profile_path'])
    report.update(total_frames=sum(r['frames'] for r in report['files']),detector_seconds=sum(r['detector_seconds'] for r in report['files']),decode_and_detect_seconds=sum(r['decode_and_detect_seconds'] for r in report['files']),session_init_seconds=init_s,warmup_seconds=warmup_s,completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    report['profiling_enabled']=bool(session) and not args.no_profile
    report['decoder']=args.decoder
    dest=root/'reports';dest.mkdir(exist_ok=True);(dest/(args.mode+('_unprofiled' if args.no_profile else '')+('_ffmpeg' if args.decoder=='ffmpeg' else '')+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('PILOT_FAILED: '+repr(e),file=sys.stderr,flush=True);raise
