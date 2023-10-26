# 3D-Pose-and-Shape-Estimation-with-a-Camera-System

<p align="center">
  <img src="/results/downtown_sitOnStairs_render.gif" data-canonical-src="/results/downtown_sitOnStairs_render.gif" width="400" />
  <img src="/results/Havoc_Ladies_render.gif" data-canonical-src="/results/Havoc_Ladies_render.gif" width="400" />
  <img src="/results/downtown_sitOnStairs_world.gif" data-canonical-src="/results/downtown_sitOnStairs_world.gif" width="400" />
  <img src="/results/Havoc_Ladies_world.gif" data-canonical-src="/results/Havoc_Ladies_world.gif" width="400" />
</p>

This repository contains the official PyTorch implementation of my master's thesis.

# Overview

In this work, a solution is addressed that try to estimate the 3D joint position of several people in in-the-wild scenes, as well as their body shape and global trajectory from a single RGB video, recorded with a static or dynamic camera.
In contrast to complex multi-view systems, this solution prioritizes simplicity and adaptability in different applications. Faced with the challenging scenario, a system was developed based on different frameworks, individually optimized for their purpose. As such, the author sought to extend the process carried out in a conventional pose and shape estimator, robustly implementing the tracking capability of humans and an inference based on temporal coherence, capable of dealing with complete occlusions over long time intervals.

The humans, present in the scene, are detected and duly identified throughout the video using an Multiple Person Tracking (MOT->[Yolo_tracking](https://github.com/mikel-brostrom/yolo_tracking)) (i.e., Deep OC-SORT with [YOLOv8x](https://github.com/ultralytics/ultralytics) and Re-ID model). This information is fed into the HPS estimator (i.e., [HybrIK](https://github.com/Jeff-sjtu/HybrIK) with backbone from the [HRNet-W48](https://drive.google.com/file/d/1gp3549vIEKfbc8SDQ-YF3Idi1aoR3DkW/view?usp=share_link) network), which is able to generate, from a combination of the volumetric representation of the joints and the ability to extract features from the DCNNs, a sequence that defines the body motion of the human in the camera’s coordinate system (i.e., root translations, root rotations, body pose and shape parameters).

In addition, the body motion, locally defined, is filled according to an iterative process, given by the integration of the generative motion optimizer, in turn organized in an architecture based on Transformers and supported by the temporal relationships present in the information of the visible detections. For a set of parameters describing the body motion of each human, the respective global trajectory is obtained, properly related, in a process based on local positional variation (position in the plane and orientation) and an iterative optimization of the camera parameters consistent with the video evidence, e.g., 2D keypoints. 

<p align="center">
<img src="https://github.com/AndreOliveira00/3D-Pose-and-Shape-Estimation-with-a-Camera-System/blob/6042793f41b5cddfee25951dd4d6f580a8892330/results/sys_architecture.png" data-canonical-src="https://github.com/AndreOliveira00/3D-Pose-and-Shape-Estimation-with-a-Camera-System/blob/6042793f41b5cddfee25951dd4d6f580a8892330/results/sys_architecture.png" width="800"/>
</p>



# Table of Content
- [Installation](#installation-instructions)
    - [Environment](#environment)
    - [Dependencies](#dependencies)
    - [Pretrained Models](#pretrained-models) 
- [Demo](#demo)
    - [Dynamic Videos](#dynamic-videos)
    - [Static Videos](#static-videos)
    - [Multi-Person Videos](#multi-person-videos)
- [Datasets](#datasets)
- [Extra Info](#extra-info)
- [Citation](#citation)

# Installation instructions
I will guide you through installing the project, however, it is recommended to read the [GLAMR](https://nvlabs.github.io/GLAMR) and [Yolo tracking](https://github.com/mikel-brostrom/yolo_tracking) installation guide.

### Environment
* **Tested OS:** MacOS, Linux (Ubuntu 20.04.6 LTS used in demo)
* Python >= 3.7 (3.9.16 used in demo)
* PyTorch >= 1.8.0 (1.9.1 used in demo)
* [HybrIK](https://github.com/Jeff-sjtu/HybrIK) (used in demo)


### Dependencies
0. The dependencies installation pipeline allows synchronization with all submodules used. It is recommended to read the installation section for each of the dependencies ([Yolo tracking](https://github.com/mikel-brostrom/yolo_tracking), [HybrIK](https://github.com/Jeff-sjtu/HybrIK) and [GLAMR](https://github.com/NVlabs/GLAMR/tree/main#demo)).
1. Clone this repository recursively:
    ```
    git clone --recursive https://github.com/AndreOliveira00/3D-Pose-and-Shape-Estimation-with-a-Camera-System.git
    ```
    This will fetch the submodules [HybrIK](https://github.com/Jeff-sjtu/HybrIK) and [Yolo tracking](https://github.com/mikel-brostrom/yolo_tracking).
    Before proceeding, please note that there are 3 anaconda environments configured and operational to import into `envs`, e.g., main_HybrIKX_env_name_hybrikx2.yaml means that it will be used in the HybrIK submodule and must have the name hybrikx2 (otherwise it must be changed in the code in [#L130](https://github.com/AndreOliveira00/3D-Pose-and-Shape-Estimation-with-a-Camera-System/blob/f582d18eff2f654365b64484167176030fa95785/pose_est/run_pose_est_demo.py#L130) 
2. Follow HybrIK's installation [instructions](https://github.com/Jeff-sjtu/HybrIK#installation-instructions) and download its [models](https://github.com/Jeff-sjtu/HybrIK#download-models).
3. Install [PyTorch 1.9.1](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version, e.g., for NVIDIA Geforce GTX 1650 run:
    ```
    conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    ```
4. Install system dependencies (Linux only, I know it might be confusing, but the main environment for GLAMR is called "**hybrik**"):
    ```
    source install.sh
    ```
5. Install python dependencies:
    ```
    pip install -r requirements.txt
    ```
6. Download [SMPL](https://smpl.is.tue.mpg.de/) models & joint regressors and place them in the `data` folder. You can obtain the model following [SPEC](https://github.com/mkocabas/SPEC)'s instructions [here](https://github.com/mkocabas/SPEC/blob/master/scripts/prepare_data.sh).

### Pretrained Models
* You can download [third-party](https://github.com/YanglanOu) pretrained models from [Google Drive](https://drive.google.com/file/d/1_3h0DExyHkPH9cv1O8Y42YPI3c08G-2b/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1nvSzfuffB5yBaZ3GRBDC5w?pwd=pj3z).
* Once the `glamr_models.zip` file is downloaded, unzipping it will create the `results` folder:
  ```
  unzip glamr_models.zip
  ```
  Note that the pretrained models directly correspond to the config files for the [motion infiller](motion_infiller/cfg) and [trajectory predictor](traj_pred/cfg).

# Demo

This section is identical to the content presented in the original project ([GLAMR](https://github.com/NVlabs/GLAMR/tree/main#demo)), however here are some guidelines for implementation.

MOT settings must be edited within the code in the MOT_settings dictionary ([#L30](https://github.com/AndreOliveira00/3D-Pose-and-Shape-Estimation-with-a-Camera-System/blob/f582d18eff2f654365b64484167176030fa95785/global_recon/run_demo.py#L30))
```python
  MOT_settings = {
    "multi_person_tracking_method": "ocsort",    # deepocsort, strongsort, botsort or sort  
    "single_person_detection_method": "ocsort",  # deepocsort, strongsort, botsort, sort  or fasterrcnn_resnet50
    "YOLO_model": "yolov8x.pt",
    "ReID_model": "osnet_ain_x1_0_msmt17.pt",
    "iou_value": 0.3,
    "conf_value": 0.6
}
```

### Dynamic Videos
Run the following command to test GLAMR on a single-person video with **dynamic** camera:
```
python global_recon/run_demo.py --cfg glamr_dynamic \
                                --video_path assets/dynamic/running.mp4 \
                                --out_dir out/glamr_dynamic/running \
                                --network_type hrnet48 \
                                --gpu 0\
                                --render_mode shape+pose\
                                --rend_fps 30\
                                --save_video
```
This will output results to `out/glamr_dynamic/running`. Results videos will be saved to `out/glamr_dynamic/running/grecon_videos`. Additional dynamic test videos can be found in [assets/dynamic](assets/dynamic).


### Static Videos
Run the following command to test GLAMR on a single-person video with **static** camera:
```
python global_recon/run_demo.py --cfg glamr_static \
                                --video_path assets/static/basketball.mp4 \
                                --out_dir out/glamr_static/basketball \
                                --network_type hrnet48 \
                                --gpu 0\
                                --render_mode shape+pose\
                                --rend_fps 30\
                                --save_video
```
This will output results to `out/glamr_static/basketball`. Results videos will be saved to `out/glamr_static/basketball/grecon_videos`. Additional static test videos can be found in [assets/static](assets/static).

### Multi-Person Videos
Use the `--multi` flag and the `glamr_static_multi` config in the above demos to test GLAMR on a **multi-person** video:
```
python global_recon/run_demo.py --cfg glamr_static_multi \
                                --video_path assets/static/basketball.mp4 \
                                --out_dir out/glamr_static_multi/basketball \
                                --network_type hrnet48 \
                                --gpu 0\
                                --render_mode shape+pose\
                                --rend_fps 30\
                                --save_video
                                --multi
```
This will output results to `out/glamr_static_multi/basketball`. Results videos will be saved to `out/glamr_static_multi/basketball/grecon_videos`.


# Datasets
The author of GLAMR used three datasets: [AMASS](https://amass.is.tue.mpg.de/), [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/), and Dynamic [Human3.6M](http://vision.imar.ro/human3.6m), I created a new one, called Dynamic [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/). Please download them from the official website and place them in the `dataset` folder with the following structure:
```
${GLAMR_ROOT}
|-- datasets
|   |-- 3DPW
|   |-- amass
|   |-- H36M
```

# Extra Info
1. If there is an error with the PyTorch3D library (HybrIK env), it is recommended to build from source (follow [tutorial](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/deform_source_mesh_to_target_mesh.ipynb) -> **0. Install and Import modules**)
2. [PyVista Error](https://github.com/NVlabs/GLAMR/issues/10). Test [PyVista](https://tutorial.pyvista.org/) (GLAMRenv) before starting. Follow [pyvista's example](https://docs.pyvista.org/examples/02-plot/gif.html?highlight=off_screen).
3. 

## Citation
If you found this work helpful in your research, please cite this repository.

The development was based on the following articles:

    @inproceedings{li2021hybrik,
        title={Hybrik: A hybrid analytical-neural inverse kinematics solution for 3d human pose and shape estimation},
        author={Li, Jiefeng and Xu, Chao and Chen, Zhicun and Bian, Siyuan and Yang, Lixin and Lu, Cewu},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={3383--3393},
        year={2021}
    }

    @article{li2023hybrik,
        title={HybrIK-X: Hybrid Analytical-Neural Inverse Kinematics for Whole-body Mesh Recovery},
        author={Li, Jiefeng and Bian, Siyuan and Xu, Chao and Chen, Zhicun and Yang, Lixin and Lu, Cewu},
        journal={arXiv preprint arXiv:2304.05690},
        year={2023}
    }
    
    @inproceedings{yuan2022glamr,
        title={GLAMR: Global Occlusion-Aware Human Mesh Recovery with Dynamic Cameras},
        author={Yuan, Ye and Iqbal, Umar and Molchanov, Pavlo and Kitani, Kris and Kautz, Jan},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2022}
    }




