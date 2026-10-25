import cv2,mediapipe as mp,subprocess,numpy as np
p='/nvmedata/workspace2/users/nina/batch_runs/autonomous_batch_20261011/staging/raw/DJI_20261004214634_0132_D.MP4'; w,hgt=640,360; fb=w*hgt*3
ff=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-xerror','-threads','2','-i',p,'-vf','fps=3,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2','-threads','2','-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
det=mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=2,model_complexity=0,min_detection_confidence=0.30,min_tracking_confidence=0.30)
for i in range(20):
 raw=ff.stdout.read(fb); print('READ',i,len(raw),flush=True)
 if len(raw)!=fb: break
 frame=np.frombuffer(raw,dtype=np.uint8).reshape((hgt,w,3)); r=det.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); print('DONEFRAME',i,len(r.multi_hand_landmarks or []),flush=True)
det.close(); ff.kill(); ff.wait(); print('DONE',flush=True)