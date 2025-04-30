

# Final Year Project Based on OpenPCDet Toolbox

## Introduction:

This final year project is based on the open-source 3D detection toolbox OpenPCDet. To focus on the core contributions of my work, I have removed unrelated components which are provided by OpenPCDet, such as implemention of other detection alogorthim, remaining code still works fine. The deployment documents were removed as well, but you can find these in **OpenPCDet** repository alternatively: https://github.com/open-mmlab/OpenPCDet.git.

This work does not claim originality for the base framework or attention mechanisms, but rather demonstrates their integration and adaptation in the context of point cloud-based 3D object detection.

## Core Contribution
Move to the appendix of Final report

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




