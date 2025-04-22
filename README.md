

# Final Year Project Based on OpenPCDet Toolbox

## Introduction:

This final year project is based on the open-source 3D detection toolbox OpenPCDet. To focus on the core contributions of my work, I have removed unrelated components which are provided by OpenPCDet, such as implemention of other detection alogorthim, remaining code still works fine. The deployment documents were removed as well, but you can find these in **OpenPCDet** repository alternatively: https://github.com/open-mmlab/OpenPCDet.git.

My core contributions include the re-implementation and integration of several attention modules (SENet, ECA-Net, and CBAM) adapted specifically for point cloud data, as well as architectural modifications to the backbone network and FPS calculation modules. Unless otherwise noted, all other components were inherited directly from the official OpenPCDet repository without modification.

All	reimplement modules and ideas were appropriately re-implemented following their official papers or reference PyTorch implementations. Full acknowledgements and licenses are listed at the end of this document.

This work does not claim originality for the base framework or attention mechanisms, but rather demonstrates their integration and adaptation in the context of point cloud-based 3D object detection.

## Core Contribution

**Attention Modules**:

- Add `pcdet/models/model_utils/attention_utils.py`: Include the customise implementations of SENet, ECA-Net, and CBAM-Net, which were implemented based on their respective papers and official PyTorch implementations, adapted to support point cloud data structure.
  
 - `class SEAttention(nn.Module)`:A custom implementation of SENet which can be used on both voxel-base models and pillar-base models.
 - `class ECAPFNLayer(nn.Module)`:A custom implementation of ECA-Net which can be used on pillar-base models only.
 - `class CBMAPFNLayer(nn.Module)`: A custom implementation of CBAM-Net which can be used on pillar-base models only.
 - `class SESparse3D(nn.Module)`: A custom implementation of SENet which was optimised for 3D spares convolution.
 - `class SESparse2D(nn.Module)`: A custom implementation of SENet which was optimised for 2D spares convolution, such as pillarnet. (Not used)
 - `class SE2D(nn.Module)`: Copied from [moskomule/senet.pytorch](https://github.com/moskomule/senet.pytorch) as a reference. (Not used in the final implementation)

**Backbone Modify**: 

- Modify `pcdet/models/backbones_3d/vfe/dynamic_pillar_vfe.py`:
  - Added `class SEDynamicPillarVFE(DynamicPillarVFE)`: Extend given VFE implemention and insert the SENet into it.
  - Added `class ECADynamicPillarVFE(DynamicPillarVFE)`: Extend given VFE implemention and insert the ECA-Net into it.
  - Added `class CBAMDynamicPillarVFE(DynamicPillarVFE)`: Extend given VFE implemention and insert the CBAM-Net into it.
- Modify `pcdet/models/backbones_3d/vfe/dynamic_pillar_se_vfe.py`:
  - Re-implement original `dynamic_pillar_vfe.py` that add SENet in multiple scales.
  - Implement the density-aware network base on SE attention principle.
  - Implement the pillar-size-adjustment network base on SE attention principle.
- Modify `pcdet/models/backbones_3d/spconv_backbone.py`:
  - Added `class SEVoxelResBackBone8x(VoxelResBackBone8x)`: Extend given voxel network and add SENet in multiple layers.
- Modify `pcdet/models/backbones_3d/spconv_backbone_2d.py`: (Not used in this project)
  - Added `class SEPillarRes18BackBone8x(PillarRes18BackBone8x)`: Extend given voxel network and add SENet in multiple layers. But this module isn't used for KITTI models.
  
**FPS Calculation**:

- Modify `tools/eval_utils/eval_utils.py`:
  - Implement `class InferenceTimeMeter`: Implement inference time calculation module based on Python offical document and Pytorch offical document.
  - Modify `def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):` Add the FPS log printer.
- Modify  `tools/test.py`:
  - Add parameters for inference time calculation.

**Configuration Files Makeup**:

- Base on KITTI dataset: `tools/cfgs/kitti_models`, Created manually by learning the YAML configuration format due to no suitable reference file was available.
- Base on nuScenes dataset: `tools/cfgs/nuscenes_models` (Not used in this project), rewritten from the officially provided configuration file.

**Visualisation: `demo.py`**:

- Contribution of this part wasn't used in this project due the dataset changing. I modify this part for nuScenes dataset.
- Run this part on non-GUI server require extra configuration of the server environment, such as vnc desktop.

## **Except for the contributions explicitly listed above, all other modules and code are provided by the official OpenPCDet repository without modification.**

## Acknowledge:

The following official and reference PyTorch implementations were used as guidance when re-implementing the attention modules in this project:

- **OpenPCDet**: https://github.com/open-mmlab/OpenPCDet.git
- **SENet (Official Recommended)**: https://github.com/moskomule/senet.pytorch  
- **ECA-Net (Official Implementation)**: https://github.com/BangguWu/ECANet  
- **CBAM (PyTorch Re-implementation)**: https://github.com/luuuyi/CBAM.PyTorch
- **FPS Calculation Relate Docs**:
 - Python: https://docs.python.org/3/library/time.html
 - Pytoch.cuda:
  - https://pytorch.org/docs/stable/cuda.html
  - https://glaringlee.github.io/cuda.html 

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




