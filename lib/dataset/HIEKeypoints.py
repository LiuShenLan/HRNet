from .HIEDataset import HIEDataset
from .target_generators import HeatmapGenerator
import numpy as np
import pycocotools

# define global variables
NUM_JOINTS = 14
WITH_CENTER = False

class HIEKeypoints(HIEDataset):
    def __init__(self, 
                cfg,
                dataset_name,
                remove_images_without_annotations,
                heatmap_generator,
                joints_generator,
                transforms = None
                ):
        super().__init__('data/HIE20/train')
        # super init HIEDataset
        
        self.num_scales = self._init_check(heatmap_generator, joints_generator)
        self.num_joints = NUM_JOINTS
        self.with_center = WITH_CENTER
        self.num_joints_without_center = self.num_joints - 1 \
            if self.with_center else self.num_joints
        self.scale_aware_sigma = False  
        self.base_sigma = 2.0              
        self.base_size = 256.0      
        self.int_sigma = False

        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.image
                # image_ids
            ]
        
        self.transforms = transforms
        self.heatmap_generator = heatmap_generator
        self.joints_generator = joints_generator

    
    def __getitem__(self, idx):
        
        img = super().__getitem__(idx)
        img_name = self.file_name[idx]
        anno = self.gt_keypoints[img_name]

        # mask = self.get_mask(anno)
        joints = self.get_joints(anno)
        mask = np.ones_like(joints)

        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]
        target_list = list()

        if self.transforms:
            # img, mask_list, joints_list = self.transforms(
            #     img, mask_list, joints_list
            # )
            img, _,joints_list = self.transforms(
                img, mask_list ,joints_list
            )

        for scale_id in range(self.num_scales):
            target_t = self.heatmap_generator[scale_id](joints_list[scale_id])
            joints_t = self.joints_generator[scale_id](joints_list[scale_id])

            target_list.append(target_t.astype(np.float32))
            # mask_list[scale_id] = mask_list[scale_id].astype(np.float32)
            joints_list[scale_id] = joints_t.astype(np.int32)

        # return img, target_list, mask_list, joints_list
        return img, target_list, joints_list

    def get_joints(self,anno):

        num_people = len(anno)

        if self.scale_aware_sigma:
            joints = np.zeros((num_people, self.num_joints, 4))
        else:
            joints = np.zeros((num_people, self.num_joints, 3))

        for i, per_person in enumerate(anno):

            joints[i, :self.num_joints_without_center, :2] = \
                np.array(per_person['keypoints']).reshape([-1, 2])

            joints[i, :self.num_joints_without_center, 2] = 1

            # format 

            if self.with_center:
                joints_sum = np.sum(joints[i, :-1, :2], axis=0)
                num_vis_joints = len(np.nonzero(joints[i, :-1, 2])[0])
                if num_vis_joints > 0:
                    joints[i, -1, :2] = joints_sum / num_vis_joints
                    joints[i, -1, 2] = 1
            
            return joints

    # def get_mask(self,anno):
        
    #     # change with the data, 11_json as debug
    #     img_height = 720
    #     img_width = 1280
    #     m = np.zeros((img_height,img_width))

    #     for per_person in anno:
            
    #         if True:
    #             rle = pycocotools.mask.frPyObjects(
    #                 # 'segmentation' :RLE or [polygon]
    #                 'RLE',img_height,img_width)
    #             m += pycocotools.mask.decode(rle)

        
    #     return m < 0.5

    def _init_check(self, heatmap_generator, joints_generator):
        assert isinstance(heatmap_generator, (list, tuple)), 'heatmap_generator should be a list or tuple'
        assert isinstance(joints_generator, (list, tuple)), 'joints_generator should be a list or tuple'
        assert len(heatmap_generator) == len(joints_generator), \
            'heatmap_generator and joints_generator should have same length,'\
            'got {} vs {}.'.format(
                len(heatmap_generator), len(joints_generator)
            )
        return len(heatmap_generator)
