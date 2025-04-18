

# Final Year Project Based on OpenPCDet Toolbox

## Introduction:

To better highlight my contribution of my final year project, I have removed the a part of code in **OpenPCDet** which is unrelate to this project. Only the components directly relevant to this work have been retained
The deployment documents were removed as well, but you can find these in **OpenPCDet** repository alternatively: https://github.com/open-mmlab/OpenPCDet.git

## Core Contribution

**Attention Modules**:

- Implement `pcdet/models/model_utils/attention_utils.py`: Include the implementations of SENet, ECA-Net, and CBAM-Net, which were re-implemented based on their respective papers and official PyTorch implementations, adapted to support point cloud data structure.
 - Implement `class SEAttention(nn.Module)`:SENet which can be used on both voxel-base models and pillar-base models.
 - Implement `class ECAPFNLayer(nn.Module)`: ECA-Net which can be used on pillar-base models only.
 - Implement `class CBMAPFNLayer(nn.Module)`: CBAM-Net wich can be used on pillar-base models only.
 - Implement `class SESparse3D(nn.Module)`: SENet which was optimised for 3D spares convolution.
 - Implement `class SESparse2D(nn.Module)`: SENet which was optimised for 2D spares convolution, such as pillarnet.
 - Copied from [moskomule/senet.pytorch](https://github.com/moskomule/senet.pytorch) as a reference.

**Backbone Modify**: 

- Modify `pcdet/models/backbones_3d/vfe/dynamic_pillar_vfe.py`:
  - Implement `class SEDynamicPillarVFE(DynamicPillarVFE)`: Extent given VFE implemention and insert the SENet into it.
  - Implement `class ECADynamicPillarVFE(DynamicPillarVFE)`: Extent given VFE implemention and insert the ECA-Net into it.
  - Implement `class CBAMDynamicPillarVFE(DynamicPillarVFE)`: Extent given VFE implemention and insert the CBAM-Net into it.
- Modify `pcdet/models/backbones_3d/vfe/dynamic_pillar_se_vfe.py`:
  - Re-implement original `dynamic_pillar_vfe.py` that add SENet in multiple scales.
  - Implement the density-aware networt base on SE attention principle.
  - Implement the pillar-size-adjustment network base on SE attention principle.
- Modify `pcdet/models/backbones_3d/spconv_backbone.py`:
  - Implement `class SEVoxelResBackBone8x(VoxelResBackBone8x)`: Extent given voxel network and add SENet in multiple layers.
- Modify `pcdet/models/backbones_3d/spconv_backbone_2d.py`: (Not used in this project)
  - Implement `class SEPillarRes18BackBone8x(PillarRes18BackBone8x)`: Extent given voxel network and add SENet in multiple layers. But this module isn't used for KITTI models.
  
**FPS Calculation**:

- Modify `tools/eval_utils/eval_utils.py`:
  - Implement `class InferenceTimeMeter`: Implement inference time calculation module based on Python offical document and CUDA offical document.
  - Modify `def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):` Add the FPS log printer.
- Modify  `tools/test.py`:
  - Add parameters for inference time calculation.

**Configuration Files Makeup**:

- Base on KITTI dataset: `tools/cfgs/kitti_models`, Created manually by learning the YAML configuration format due to no suitable reference file was available.
- Base on nuScenes dataset: `tools/cfgs/nuscenes_models` (Not used in this project), rewritten from the officially provided configuration file.

**Visualisation: `demo.py`**:

- Contribution of this part wasn't used in this project due the dataset changing. I modify this part for nuScenes dataset.
- Run this part on non-GUI server require extra configuration of the server environment, such as vnc desktop.

## Acknowledge:

The following official and reference PyTorch implementations were used as guidance when re-implementing the attention modules in this project:

- **OpenPCDet**: https://github.com/open-mmlab/OpenPCDet.git
- **SENet (Official Recommended)**: https://github.com/moskomule/senet.pytorch  
- **ECA-Net (Official Implementation)**: https://github.com/BangguWu/ECANet  
- **CBAM (PyTorch Re-implementation)**: https://github.com/luuuyi/CBAM.PyTorch


## License

`OpenPCDet` is released under the [Apache 2.0 license](LICENSE).

## Citation 
If you find this project useful in your research, please consider cite:


```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```




