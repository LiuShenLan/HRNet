import os
import os.path
import numpy as np
import json

from os.path import join as opj
from torch.utils.data import Dataset
from collections import defaultdict
from collections import OrderedDict
from PIL import Image

class HIEDataset(Dataset):
    def __init__(self,data_path):
        super(HIEDataset,self).__init__()
        self.root = 'data/HIE20/labels/train/track2&3'
        self.json_name = '11.json'
        self.transform = None
        self.file_name = []
        self.image = []
        self.data_path = data_path
        self.params = Params(iouType='keypoints')
        self.gt_keypoints = defaultdict(list)

        self.read_file_name()
        self.read_gts_kpts()
        self.get_images()
    
    def get_images(self):
        folders = os.listdir(self.data_path)
        for folder in folders:
            folder_path = opj(self.data_path,folder)
            images = os.listdir(folder_path)
            for image in images:
                pic_path = opj(self.data_path,folder,image)
                self.image.append(pic_path)
    
    def __getitem__(self, index):
        
        dataset = np.array(Image.open(self.image[index]))
        return dataset 

    def __len__(self):

        return len(self.image)

    def read_json(self,json_name):
        with open(json_name) as file:
            content = json.load(file)
        return content

    def read_file_name(self):
        json_name = opj(self.root,self.json_name)
        content = self.read_json(json_name)
        annolist = content['annolist']
        for per_annolist in annolist:
            
            # extract image 
            image = per_annolist['image']
            # image_name as the key to get information
            # image_name format 000000.jpg
            image_name = image[0]['name']
            self.file_name.append(image_name)

        return self.file_name
    
    def read_gts_kpts(self):
        
        json_name = opj(self.root,self.json_name)
        content = self.read_json(json_name)
        annolist = content['annolist']
        for per_annolist in annolist:

            # extract image name as key 
            image = per_annolist['image']
            image_name = image[0]['name']

            for per_person in per_annolist['annorect']:
                
                # key_points = []
                key_points = np.zeros((14,2),dtype=float)
                annopoints = per_person['annopoints'][0]['point']
                for per_joint in annopoints:
                    ids = per_joint['id'][0]
                    # per_point = [per_joint['x'][0],per_joint['y'][0]]
                    # key_points.append(per_point)
                    key_points[ids,0] = per_joint['x'][0]
                    key_points[ids,1] = per_joint['y'][0]
                key_points = key_points.tolist()

                bbox = [per_person['x1'][0],per_person['y1'][0],
                        per_person['x2'][0],per_person['y2'][0]]
                self.gt_keypoints[image_name].append({
                    'bbox':bbox,
                    'score':per_person['score'][0],
                    'track_ids':per_person['track_id'][0],
                    'keypoints':key_points # joints x 2(x,y)
                })
        
        return self.gt_keypoints



    def evaluate(self, cfg, preds, scores, output_dir,
                 *args, **kwargs):

        # save result        
        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            # res_folder, 'keypoints_%s_results.json' % self.dataset)
            res_folder, 'keypoints_results.json')
        # preds is a list of: image x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)

        kpts = defaultdict(list)
        for idx, _kpts in enumerate(preds):
            file_name = self.file_name[idx]
            for idx_kpt, kpt in enumerate(_kpts):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self.processKeypoints(kpt)
                # if self.with_center: False

                # 000000.jpg [-10:-4]
                # kpts[int(file_name[-10:-4])].append(
                kpts[file_name].append(
                    {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        'tags': kpt[:, 3],
                        'image': int(file_name[-10:-4]),
                        'area': area
                    }
                )
        
        # rescoring and oks nms
        oks_nmsed_kpts = []
        # image x person x (keypoints)
        for img in kpts.keys():
            # person x (keypoints)
            img_kpts = kpts[img]
            # person x (keypoints)
            # do not use nms, keep all detections
            keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        # if test in dataset
        if True:
            # info_str = self.do_keypoint_eval(
            #     res_file, res_folder
            # )
            # name_value = OrderedDict(info_str)
            # return name_value, name_value['AP']
            for image_name in self.file_name:
                computed_oks = self.do_computedOKS(self.gt_keypoints,kpts,image_name)
                print(computed_oks)
        else:
            return {'Null': 0}, 0


    def processKeypoints(self, keypoints):

        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            p = keypoints[keypoints[:, 2] > 0][:, :2].mean(axis=0)
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]

        return tmp
    
    def process_dt(self,xd):
        # xd.shape (17,)
        xd[0],xd[1],xd[2],xd[3],xd[4],xd[5],xd[6],
        xd[7],xd[8],xd[9],xd[10],xd[11],xd[12],xd[13]=\
        xd[0],xd[0],xd[5],xd[7],xd[9],xd[6],xd[8],
        xd[10],xd[11],xd[13],xd[15],xd[12],xd[14],xd[16]
        
        xd = xd[0:14]
        # extract xd[0]-xd[13]
        return xd
    
    def do_keypoint_eval(self, res_file, res_folder):
        pass

    def do_computedOKS(self, gts, pred,image_name):

        p = self.params
        # dimention here should be Nxm
        gts_all = gts 
        dts_all = pred
        gts = gts_all[image_name]
        dts = dts_all[image_name]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        # dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        
        for i, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            # i is the index of person
            g = np.array(gt['keypoints'])
            print('g',g.shape)
            xg = g[:,0]; yg = g[:,1]
            print('xg',xg.shape)
            print('yg',yg.shape)
            k1 = 14     # visible joint_num
            bb = gt['bbox']
            x0 = bb[0]; x1 = bb[0] + (bb[2]-bb[0]) * 2
            y0 = bb[1]; y1 = bb[1] + (bb[3]-bb[1]) * 2
            print('bbox:',x0,y0,x1,y1)

            for j,dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[:,0]; yd = d[:,1]
                xd = self.process_dt(xd)
                yd = self.process_dt(yd)
                print('d:',d.shape)
                tag = dt['tags']

                if k1>0:
                # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (dt['area']+np.spacing(1)) / 2
                # gt['area'] => dt['area] as test
                # if k1 > 0:
                #     e=e[vg > 0]
                ious[j,i] = np.sum(np.exp(-e)) / e.shape[0]
            print(ious)
        return ious

    def save_json(self,preds,scores):

        kpts = defaultdict(list)
        for idx, _kpts in enumerate(preds):
            file_name = self.file_name[idx]
            for idx_kpt, kpt in enumerate(_kpts):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self.processKeypoints(kpt)
                # if self.with_center: False

                # 000000.jpg [-10:-4]
                # kpts[int(file_name[-10:-4])].append(
                kpts[file_name].append(
                    {
                        'keypoints': kpt[:, 0:3],
                        'score': scores[idx][idx_kpt],
                        'tags': kpt[:, 3],
                        'image': int(file_name[-10:-4]),
                        'area': area
                    }
                )
        
        # construct json dict
        annolist_list = []
        for image_name in kpts.keys():
            
            # annolist_dict = defaultdict(list)
            # image_name_dict = defaultdict(list)
            annolist_dict = {}
            image_name_dict = {}
            image_name_dict['name'] = image_name
            annorect_list = []

            for idx,per_person in enumerate(kpts[image_name]):

                # annorect_dict = defaultdict(list)
                annorect_dict = {}
                annorect_dict['x1'] = [np.min(per_person['keypoints'][:,0])]
                annorect_dict['y1'] = [np.min(per_person['keypoints'][:,1])]
                annorect_dict['x2'] = [np.max(per_person['keypoints'][:,0])]
                annorect_dict['y2'] = [np.max(per_person['keypoints'][:,1])]
                annorect_dict['track_id'] = [idx]
                # construct point dict
                # 17 point
                point_list = []
                for i in range(17):
                    # per_point_dict = defaultdict(list)
                    per_point_dict = {}
                    per_point_dict['id'] = [i]
                    per_point_dict['x'] = [per_person['keypoints'][i,0]]
                    per_point_dict['y'] = [per_person['keypoints'][i,1]]
                    per_point_dict['score'] = [per_person['score']]
                    point_list.append(per_point_dict)
                # point_dict = defaultdict(list)
                point_dict = {}
                point_dict['point'] = point_list

                annorect_dict['annopoints'] = [point_dict]

                annorect_list.append(annorect_dict)

            annolist_dict['image'] = [image_name_dict]
            annolist_dict['ignore_regions'] = []
            annolist_dict['annorect'] = annorect_list

            annolist_list.append(annolist_dict)
        
        # all_annolist_dict = defaultdict(list)
        all_annolist_dict = {}
        all_annolist_dict['annolist'] = annolist_list
        # write json file
        # jsondata = json.dumps(all_annolist_dict,indent=4,separators=(',', ': '),cls=MyEncoder)
        jsondata = json.dumps(all_annolist_dict,cls=MyEncoder)
        f = open('11_result.json', 'w')
        f.write(jsondata)
        f.close()
 
class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [50]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87])/10.0

    def __init__(self, iouType='keypoints'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
