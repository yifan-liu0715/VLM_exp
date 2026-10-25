# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Extracted from make_openpose/preprocess/pose2d.py and pose2d_utils.py
import os
import cv2
import warnings
from typing import Union, List
import numpy as np
import torch
import onnxruntime


def box_convert_simple(box, convert_type='xyxy2xywh'):
    if convert_type == 'xyxy2xywh':
        return [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    elif convert_type == 'xywh2xyxy':
        return [box[0], box[1], box[2] + box[0], box[3] + box[1]]
    elif convert_type == 'xyxy2ctwh':
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]]
    elif convert_type == 'ctwh2xyxy':
        return [box[0] - box[2] // 2, box[1] - box[3] // 2, box[0] + (box[2] - box[2] // 2), box[1] + (box[3] - box[3] // 2)]


def read_img(image, convert='RGB', check_exist=False):
    if isinstance(image, str):
        from os import path as osp
        if check_exist and not osp.exists(image):
            return None
        try:
            from PIL import Image
            img = Image.open(image)
            if convert:
                img = img.convert(convert)
        except:
            raise IOError('File error: ', image)
        return np.asarray(img)
    else:
        if isinstance(image, np.ndarray):
            # Do NOT flip channels here; assume caller passes correct format (RGB)
            return image
        else:
            if convert:
                img = img.convert(convert)
            return np.asarray(img)


def transform_preds(coords, center, scale, output_size, use_udp=False):
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords


def get_transform(center, scale, res, rot=0):
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1


def bbox_from_detector(bbox, input_resolution=(224, 224), rescale=1.25):
    CROP_IMG_HEIGHT, CROP_IMG_WIDTH = input_resolution
    CROP_ASPECT_RATIO = CROP_IMG_HEIGHT / float(CROP_IMG_WIDTH)

    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    center = np.array([center_x, center_y])

    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_size = max(bbox_w * CROP_ASPECT_RATIO, bbox_h)

    scale = np.array([bbox_size / CROP_ASPECT_RATIO, bbox_size]) / 200.0
    scale *= rescale
    return center, scale


def crop(img, center, scale, res):
    ul = np.array(transform([1, 1], center, max(scale), res, invert=1)) - 1
    br = np.array(transform([res[1] + 1, res[0] + 1], center, max(scale), res, invert=1)) - 1

    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)

    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    try:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    except Exception as e:
        print(e)

    new_img = cv2.resize(new_img, (res[1], res[0]))
    return new_img, new_shape, (old_x, old_y), (new_x, new_y)


def _get_max_preds(heatmaps):
    assert isinstance(heatmaps, np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _taylor(heatmap, coord):
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (
            heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
            heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
            heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def _gaussian_blur(heatmaps, kernel=11):
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def keypoints_from_heatmaps(heatmaps,
                            center,
                            scale,
                            unbiased=False,
                            post_process='default',
                            kernel=11,
                            valid_radius_factor=0.0546875,
                            use_udp=False,
                            target_type='GaussianHeatmap'):
    heatmaps = heatmaps.copy()

    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'

    if post_process is False:
        warnings.warn(
            'post_process=False is deprecated, '
            'please use post_process=None instead', DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                'post_process=True, unbiased=True is deprecated,'
                " please use post_process='unbiased' instead",
                DeprecationWarning)
            post_process = 'unbiased'
        else:
            warnings.warn(
                'post_process=True, unbiased=False is deprecated, '
                "please use post_process='default' instead",
                DeprecationWarning)
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            warnings.warn(
                'unbiased=True is deprecated, please use '
                "post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'

    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        if target_type.lower() == 'GaussianHeatMap'.lower():
            preds, maxvals = _get_max_preds(heatmaps)
            # post_dark_udp is not copied; this branch requires use_udp=True which is not used by ViTPose
            raise NotImplementedError("use_udp=True is not supported in this extracted version")
        elif target_type.lower() == 'CombinedTarget'.lower():
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == 'unbiased':
            heatmaps = np.log(
                np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5

    for i in range(N):
        preds[i] = transform_preds(
            preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals


def split_kp2ds_for_aa(kp2ds, ret_face=False):
    kp2ds_body = (kp2ds[[0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 20]] + kp2ds[[0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 21]]) / 2
    kp2ds_lhand = kp2ds[91:112]
    kp2ds_rhand = kp2ds[112:133]
    kp2ds_face = kp2ds[22:91]
    if ret_face:
        return kp2ds_body.copy(), kp2ds_lhand.copy(), kp2ds_rhand.copy(), kp2ds_face.copy()
    return kp2ds_body.copy(), kp2ds_lhand.copy(), kp2ds_rhand.copy()


def load_pose_metas_from_kp2ds_seq(kp2ds_seq, width, height):
    metas = []
    for kps in kp2ds_seq:
        kps = kps.copy()
        kps[:, 0] /= width
        kps[:, 1] /= height
        kp2ds_body, kp2ds_lhand, kp2ds_rhand, kp2ds_face = split_kp2ds_for_aa(kps, ret_face=True)

        if kp2ds_body[:, :2].min(axis=1).max() < 0:
            kp2ds_body = last_kp2ds_body
        last_kp2ds_body = kp2ds_body

        meta = {
            "width": width,
            "height": height,
            "keypoints_body": kp2ds_body,
            "keypoints_left_hand": kp2ds_lhand,
            "keypoints_right_hand": kp2ds_rhand,
            "keypoints_face": kp2ds_face,
        }
        metas.append(meta)
    return metas


class SimpleOnnxInference(object):
    def __init__(self, checkpoint, device='cuda', reverse_input=False, **kwargs):
        if isinstance(device, str):
            device = torch.device(device)
        # In distributed training, 'cuda' without index defaults to cuda:0 which
        # causes all ranks to collide on the same GPU. Map to current device instead.
        if device.type == 'cuda' and device.index is None:
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{torch.cuda.current_device()}')
            else:
                device = torch.device('cpu')
        if device.type == 'cuda':
            device_id = str(device.index)
            providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.device = device
        if not os.path.exists(checkpoint):
            raise RuntimeError("{} is not existed!".format(checkpoint))
        
        if os.path.isdir(checkpoint):
            checkpoint = os.path.join(checkpoint, 'end2end.onnx')

        self.session = onnxruntime.InferenceSession(checkpoint,
                                                    providers=providers
                                                    )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_resolution = self.session.get_inputs()[0].shape[2:] if not reverse_input else self.session.get_inputs()[0].shape[2:][::-1]
        self.input_resolution = np.array(self.input_resolution)
        

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

    def get_output_names(self):
        output_names = []
        for node in self.session.get_outputs():
            output_names.append(node.name)
        return output_names


    def set_device(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cuda':
            device_id = str(device.index) if device.index is not None else "0"
            providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session.set_providers(providers)
        self.device = device


class Yolo(SimpleOnnxInference):
    def __init__(self, checkpoint, device='cuda', threshold_conf=0.05, threshold_multi_persons=0.1, input_resolution=(640, 640), threshold_iou=0.5, threshold_bbox_shape_ratio=0.4, cat_id=[1], select_type='max', strict=True, sorted_func=None, **kwargs):
        super(Yolo, self).__init__(checkpoint, device=device, **kwargs)
        
        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape

        self.input_width = 640
        self.input_height = 640
        
        self.threshold_multi_persons = threshold_multi_persons
        self.threshold_conf = threshold_conf
        self.threshold_iou = threshold_iou
        self.threshold_bbox_shape_ratio = threshold_bbox_shape_ratio
        self.input_resolution = input_resolution
        self.cat_id = cat_id
        self.select_type = select_type
        self.strict = strict
        self.sorted_func = sorted_func
        
        
    def preprocess(self, input_image):
        img = read_img(input_image)
        img_height, img_width = img.shape[:2]
        img = cv2.resize(img, (self.input_resolution[1], self.input_resolution[0]))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = image_data.astype(np.float32)
        return image_data, np.array([img_height, img_width])

    
    def postprocess(self, output, shape_raw, cat_id=[1]):
        outputs = np.squeeze(output)
        if len(outputs.shape) == 1:
            outputs = outputs[None]
        if output.shape[-1] != 6 and output.shape[1] == 84:
            outputs = np.transpose(outputs)
        
        rows = outputs.shape[0]

        x_factor = shape_raw[1] / self.input_width
        y_factor = shape_raw[0] / self.input_height

        boxes = []
        scores = []
        class_ids = []

        if outputs.shape[-1] == 6:
            max_scores = outputs[:, 4]
            classid = outputs[:, -1]
            
            threshold_conf_masks = max_scores >= self.threshold_conf
            classid_masks = classid[threshold_conf_masks] != 3.14159

            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[threshold_conf_masks][classid_masks]

            boxes = outputs[:, :4][threshold_conf_masks][classid_masks]
            boxes[:, [0, 2]] *= x_factor
            boxes[:, [1, 3]] *= y_factor
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            boxes = boxes.astype(np.int32)

        else:
            classes_scores = outputs[:, 4:]
            max_scores = np.amax(classes_scores, -1)
            threshold_conf_masks = max_scores >= self.threshold_conf

            classid = np.argmax(classes_scores[threshold_conf_masks], -1)

            classid_masks = classid!=3.14159
            
            classes_scores = classes_scores[threshold_conf_masks][classid_masks]
            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[classid_masks]
    
            xywh = outputs[:, :4][threshold_conf_masks][classid_masks]

            x = xywh[:, 0:1]
            y = xywh[:, 1:2]
            w = xywh[:, 2:3]
            h = xywh[:, 3:4]
    
            left = ((x - w / 2) * x_factor)
            top = ((y - h / 2) * y_factor)
            width = (w * x_factor)
            height = (h * y_factor)
            boxes = np.concatenate([left, top, width, height], axis=-1).astype(np.int32)

        boxes = boxes.tolist()
        scores = max_scores.tolist()
        class_ids = classid.tolist()

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.threshold_conf, self.threshold_iou)
        
        results = []
        for i in indices:
            box = box_convert_simple(boxes[i], 'xywh2xyxy')
            score = scores[i]
            class_id = class_ids[i]
            results.append(box + [score] + [class_id])

        return np.array(results)

    
    def process_results(self, results, shape_raw, cat_id=[1], single_person=True):
        if isinstance(results, tuple):
            det_results = results[0]
        else:
            det_results = results

        person_results = []
        person_count = 0
        if len(results):
            max_idx = -1
            max_bbox_size = shape_raw[0] * shape_raw[1] * -10
            max_bbox_shape = -1
            
            bboxes = []
            idx_list = []
            for i in range(results.shape[0]):
                bbox = results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    idx_list.append(i)
                    bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                    if bbox_shape > max_bbox_shape:
                        max_bbox_shape = bbox_shape
            
            results = results[idx_list]

            for i in range(results.shape[0]):
                bbox = results[i]
                bboxes.append(bbox)
                if self.select_type == 'max':
                    bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1
                bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                if bbox_size > max_bbox_size:
                    if (self.strict or max_idx != -1) and bbox_shape < max_bbox_shape * self.threshold_bbox_shape_ratio:
                        continue
                    max_bbox_size = bbox_size
                    max_bbox_shape = bbox_shape
                    max_idx = i

            if self.sorted_func is not None and len(bboxes) > 0:
                max_idx = self.sorted_func(bboxes, shape_raw)
                bbox = bboxes[max_idx]
                if self.select_type == 'max':
                    max_bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    max_bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1
                
            if max_idx != -1:
                person_count = 1

            if max_idx != -1:
                person = {}
                person['bbox'] = results[max_idx, :5]
                person['track_id'] = int(0)
                person_results.append(person)

            for i in range(results.shape[0]):
                bbox = results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    if self.select_type == 'max':
                        bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                    elif self.select_type == 'center':
                        bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1
                    if i != max_idx and bbox_size > max_bbox_size * self.threshold_multi_persons and bbox_size < max_bbox_size:
                        person_count += 1
                        if not single_person:
                            person = {}
                            person['bbox'] = results[i, :5]
                            person['track_id'] = int(person_count - 1)
                            person_results.append(person)                   
            return person_results
        else:
            return None
        

    def postprocess_threading(self, outputs, shape_raw, person_results, i, single_person=True, **kwargs):
        result = self.postprocess(outputs[i], shape_raw[i], cat_id=self.cat_id)
        result = self.process_results(result, shape_raw[i], cat_id=self.cat_id, single_person=single_person)
        if result is not None and len(result) != 0:
            person_results[i] = result


    def forward(self, img, shape_raw, **kwargs):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            shape_raw = shape_raw.cpu().numpy()

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img})[0]
        person_results = [[{'bbox': np.array([0., 0., 1.*shape_raw[i][1], 1.*shape_raw[i][0], -1]), 'track_id': -1}] for i in range(len(outputs))]

        for i in range(len(outputs)):
            self.postprocess_threading(outputs, shape_raw, person_results, i, **kwargs)         
        return person_results


class ViTPose(SimpleOnnxInference):
    def __init__(self, checkpoint, device='cuda', **kwargs):
        super(ViTPose, self).__init__(checkpoint, device=device)

    def forward(self, img, center, scale, **kwargs):
        heatmaps = self.session.run([], {self.session.get_inputs()[0].name: img})[0]
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                            center=center,
                                            scale=scale*200,
                                            unbiased=True, 
                                            use_udp=False)
        return np.concatenate([points, prob], axis=2)


    @staticmethod
    def preprocess(img, bbox=None, input_resolution=(256, 192), rescale=1.25, mask=None, **kwargs):
        if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
            bbox = np.array([0, 0, img.shape[1], img.shape[0]])
        
        bbox_xywh = bbox
        if mask is not None:
            img = np.where(mask>128, img, mask)

        if isinstance(input_resolution, int):
            center, scale = bbox_from_detector(bbox_xywh, (input_resolution, input_resolution), rescale=rescale)
            img, new_shape, old_xy, new_xy = crop(img, center, scale, (input_resolution, input_resolution))
        else:
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img, new_shape, old_xy, new_xy = crop(img, center, scale, (input_resolution[0], input_resolution[1]))

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        img_norm = (img / 255. - IMG_NORM_MEAN) / IMG_NORM_STD
        img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)
        return img_norm, np.array(center), np.array(scale)


class Pose2d:
    def __init__(self, checkpoint, detector_checkpoint=None, device='cuda', **kwargs):

        if detector_checkpoint is not None:
            self.detector = Yolo(detector_checkpoint, device)
        else:
            self.detector = None

        self.model = ViTPose(checkpoint, device)
        self.device = device

    def load_images(self, inputs):
        if isinstance(inputs, str):
            if inputs.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                cap = cv2.VideoCapture(inputs)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                images = frames
            elif inputs.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img = cv2.cvtColor(cv2.imread(inputs), cv2.COLOR_BGR2RGB)
                if img is None:
                    raise ValueError(f"Cannot read image: {inputs}")
                images = [img]
            else:
                raise ValueError(f"Unsupported file format: {inputs}")
                
        elif isinstance(inputs, np.ndarray):
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in inputs]
        elif isinstance(inputs, list):
            # Assume inputs are already RGB numpy arrays
            images = inputs
        return images

    def __call__(
        self, 
        inputs: Union[str, np.ndarray, List[np.ndarray]],
        return_image: bool = False,
        **kwargs
    ):
        images = self.load_images(inputs)
        H, W = images[0].shape[:2]
        if self.detector is not None:
            bboxes = []
            for _image in images:
                img, shape = self.detector.preprocess(_image)
                bboxes.append(self.detector(img[None], shape[None])[0][0]["bbox"])
        else:
            bboxes = [None] * len(images)

        kp2ds = []
        for _image, _bbox in zip(images, bboxes):
            img, center, scale = self.model.preprocess(_image, _bbox)
            kp2ds.append(self.model(img[None], center[None], scale[None]))
        kp2ds = np.concatenate(kp2ds, 0)
        metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)
        return metas


def extract_pose_keypoints_from_video(video_path, pose_checkpoint, detector_checkpoint=None, device='cuda'):
    """
    Extract pose keypoints from a video file path.
    
    Args:
        video_path: Path to video file.
        pose_checkpoint: Path to ViTPose ONNX checkpoint.
        detector_checkpoint: Path to YOLO detector ONNX checkpoint (optional).
        device: 'cuda' or 'cpu'.
    
    Returns:
        List of pose meta dicts, one per frame.
    """
    pose2d = Pose2d(checkpoint=pose_checkpoint, detector_checkpoint=detector_checkpoint, device=device)
    metas = pose2d(video_path)
    return metas


def extract_pose_keypoints_from_frames(frames, pose_checkpoint, detector_checkpoint=None, device='cuda'):
    """
    Extract pose keypoints from a list of RGB frames.
    
    Args:
        frames: List of RGB numpy arrays with shape (H, W, 3).
        pose_checkpoint: Path to ViTPose ONNX checkpoint.
        detector_checkpoint: Path to YOLO detector ONNX checkpoint (optional).
        device: 'cuda' or 'cpu'.
    
    Returns:
        List of pose meta dicts, one per frame.
    """
    pose2d = Pose2d(checkpoint=pose_checkpoint, detector_checkpoint=detector_checkpoint, device=device)
    metas = pose2d(frames)
    return metas
