#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/workers/base/openpcdet/__init__.py
# @IDE     : vscode

# import argparse
# import glob
# from pathlib import Path
# import numpy as np
# import torch

# from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.datasets import DatasetTemplate
# from pcdet.models import build_network, load_data_to_gpu
# from pcdet.utils import common_utils


# class DemoDataset(DatasetTemplate):
#     def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
#         """
#         Args:
#             root_path:
#             dataset_cfg:
#             class_names:
#             training:
#             logger:
#         """
#         super().__init__(
#             dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
#         )
#         self.root_path = root_path
#         self.ext = ext
#         data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

#         data_file_list.sort()
#         self.sample_file_list = data_file_list

#     def __len__(self):
#         return len(self.sample_file_list)

#     def __getitem__(self, index):
#         file_postfix = self.sample_file_list[index].as_posix().split(".")[-1]
#         if file_postfix == 'bin':
#             points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
#         elif file_postfix == 'npy':
#             points = np.load(self.sample_file_list[index])
#         elif file_postfix == 'pcd':
#             print("\033[93mPlease run the pcd2bin file first!\033[0m Check the file in this folder, \n\ror click here: https://github.com/Kin-Zhang/OpenPCDet/blob/master/tools/pcd2bin.py \n\r")
#             raise NotImplementedError
#         else:
#             raise NotImplementedError

#         input_dict = {
#             'points': points,
#             'frame_id': index,
#         }

#         data_dict = self.prepare_data(data_dict=input_dict)
#         return data_dict


# def parse_config():
#     parser = argparse.ArgumentParser(description='arg parser')
#     parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
#                         help='specify the config for demo')
#     parser.add_argument('--data_path', type=str, default='demo_data',
#                         help='specify the point cloud data file or directory')
#     parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
#     parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

#     args = parser.parse_args()

#     cfg_from_yaml_file(args.cfg_file, cfg)

#     return args, cfg


# def main():
#     args, cfg = parse_config()
#     logger = common_utils.create_logger()
#     demo_dataset = DemoDataset(
#         dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
#         root_path=Path(args.data_path), ext=args.ext, logger=logger
#     )
#     logger.info(f'Total number of samples: \t{len(demo_dataset)}')

#     model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
#     model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
#     model.cuda()
#     model.eval()
#     with torch.no_grad():
#         for idx, data_dict in enumerate(demo_dataset):
#             logger.info(f'Visualized sample index: \t{idx + 1}')
#             data_dict = demo_dataset.collate_batch([data_dict])
#             load_data_to_gpu(data_dict)
#             pred_dicts, _ = model.forward(data_dict)

#     logger.info('Demo done.')




# if __name__ == '__main__':
#     main()
import glob
from pathlib import Path
import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import logging
from typing import Union, List, Dict, Optional


class DemoDataset(DatasetTemplate):
    """演示数据集类，用于加载单帧或多帧点云数据"""
    
    def __init__(self, 
                 dataset_cfg, 
                 class_names, 
                 root_path: Union[str, Path], 
                 ext: str = '.bin', 
                 training: bool = False, 
                 logger=None):
        """
        Args:
            dataset_cfg: 数据集配置
            class_names: 类别名称列表
            root_path: 点云文件路径（文件或目录）
            ext: 文件扩展名（.bin, .npy, .pcd）
            training: 是否为训练模式
            logger: 日志记录器
        """
        super().__init__(
            dataset_cfg=dataset_cfg, 
            class_names=class_names, 
            training=training, 
            root_path=Path(root_path), 
            logger=logger
        )
        
        self.root_path = Path(root_path)
        self.ext = ext
        
        # 获取所有点云文件
        if self.root_path.is_dir():
            data_file_list = glob.glob(str(self.root_path / f'*{self.ext}'))
        else:
            data_file_list = [self.root_path]
        
        data_file_list.sort()
        self.sample_file_list = data_file_list
        
        # 添加pcd格式支持
        self.supported_formats = {
            '.bin': self._load_bin,
            '.npy': self._load_npy,
            '.pcd': self._load_pcd,
        }
    
    def __len__(self):
        return len(self.sample_file_list)
    
    def _load_bin(self, file_path):
        """加载.bin格式点云"""
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    
    def _load_npy(self, file_path):
        """加载.npy格式点云"""
        return np.load(file_path)
    
    def _load_pcd(self, file_path):
        """加载.pcd格式点云（需要安装open3d）"""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points, dtype=np.float32)
            
            # 如果有强度信息，添加到第4列
            if pcd.has_colors():
                colors = np.asarray(pcd.colors, dtype=np.float32)
                intensity = np.mean(colors, axis=1, keepdims=True)
                points = np.hstack([points, intensity])
            else:
                # 如果没有强度，填充1.0
                intensity = np.ones((points.shape[0], 1), dtype=np.float32)
                points = np.hstack([points, intensity])
            
            return points
        except ImportError:
            raise ImportError("Please install open3d to load pcd files: pip install open3d")
    
    def __getitem__(self, index):
        """获取单个样本"""
        file_path = self.sample_file_list[index]
        file_ext = Path(file_path).suffix.lower()
        
        # 根据扩展名选择加载方法
        if file_ext in self.supported_formats:
            points = self.supported_formats[file_ext](file_path)
        else:
            raise NotImplementedError(f"Unsupported file extension: {file_ext}")
        
        input_dict = {
            'points': points,
            'frame_id': index,
        }
        
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def load_model(config_path: str, 
               ckpt_path: str, 
               class_names: Optional[List[str]] = None,
               device: str = 'cuda'):
    """
    加载OpenPCDet模型
    
    Args:
        config_path: 配置文件路径（.yaml）
        ckpt_path: 预训练权重路径
        class_names: 类别名称列表（如果为None，则从配置文件中读取）
        device: 运行设备（'cuda' 或 'cpu'）
    
    Returns:
        model: 加载好的模型
        cfg: 配置对象
    """
    # 加载配置
    cfg_from_yaml_file(config_path, cfg)
    
    # 设置类别名称
    if class_names is not None:
        cfg.CLASS_NAMES = class_names
    
    # 创建logger
    logger = common_utils.create_logger()
    logger.info(f'Loading model from {ckpt_path}')
    
    # 构建模型
    model = build_network(
        model_cfg=cfg.MODEL, 
        num_class=len(cfg.CLASS_NAMES), 
        dataset=None  # 临时设为None，加载权重后再设置
    )
    
    # 加载权重
    model.load_params_from_file(
        filename=ckpt_path, 
        logger=logger, 
        to_cpu=(device == 'cpu')
    )
    
    # 移动到指定设备
    model = model.to(device)
    model.eval()
    
    return model, cfg, logger


def infer_single_frame(model, 
                       points: np.ndarray, 
                       cfg, 
                       class_names: List[str],
                       device: str = 'cuda') -> Dict:
    """
    对单帧点云进行推理
    
    Args:
        model: 加载好的模型
        points: 点云数据，shape (N, 4) 或 (N, 3)
        cfg: 配置对象
        class_names: 类别名称列表
        device: 运行设备
    
    Returns:
        pred_dicts: 预测结果字典
    """
    # 确保点云有4列（x, y, z, intensity）
    if points.shape[1] == 3:
        intensity = np.ones((points.shape[0], 1), dtype=np.float32)
        points = np.hstack([points, intensity])
    
    # 创建临时数据集
    temp_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=class_names,
        root_path='.',  # 临时路径
        training=False
    )
    
    # 准备数据字典
    input_dict = {
        'points': points,
        'frame_id': 0,
    }
    data_dict = temp_dataset.prepare_data(data_dict=input_dict)
    
    # collate batch
    data_dict = temp_dataset.collate_batch([data_dict])
    
    # 移动到GPU
    load_data_to_gpu(data_dict)
    
    # 推理
    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)
    
    return pred_dicts[0]  # 返回第一帧（也是唯一一帧）的结果


def openpcdet_inference(data_path: Union[str, Path],
                        config_path: str,
                        ckpt_path: str,
                        class_names: Optional[List[str]] = None,
                        ext: str = '.bin',
                        device: str = 'cuda',
                        return_all_frames: bool = False) -> Union[Dict, List[Dict]]:
    """
    OpenPCDet推理主函数
    
    Args:
        data_path: 点云数据路径（可以是文件或目录）
        config_path: 配置文件路径
        ckpt_path: 预训练权重路径
        class_names: 类别名称列表（如果为None，则从配置文件读取）
        ext: 文件扩展名（.bin, .npy, .pcd）
        device: 运行设备
        return_all_frames: 是否返回所有帧的结果（如果data_path是目录）
    
    Returns:
        单帧模式：返回单个预测结果字典
        多帧模式：返回预测结果列表，每帧一个字典
        每个字典包含：
            - 'pred_boxes': 预测框 [N, 7] (x, y, z, l, w, h, yaw)
            - 'pred_scores': 预测分数 [N]
            - 'pred_labels': 预测标签 [N]
    """
    # 1. 加载模型和配置
    model, cfg, logger = load_model(config_path, ckpt_path, class_names, device)
    
    # 2. 创建数据集
    dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=data_path,
        ext=ext,
        logger=logger
    )
    
    logger.info(f'Total number of samples: {len(dataset)}')
    
    # 3. 推理所有帧
    all_predictions = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            logger.info(f'Processing sample {idx + 1}/{len(dataset)}')
            
            # 获取单帧数据
            data_dict = dataset[idx]
            
            # collate batch（保持batch维度）
            data_dict = dataset.collate_batch([data_dict])
            
            # 移动到GPU
            load_data_to_gpu(data_dict)
            
            # 推理
            pred_dicts, _ = model.forward(data_dict)
            
            # 处理结果
            pred_dict = pred_dicts[0]  # 取出batch中的第一帧
            
            # 转换为numpy并添加到列表
            frame_result = {
                'frame_id': idx,
                'pred_boxes': pred_dict['pred_boxes'].cpu().numpy(),
                'pred_scores': pred_dict['pred_scores'].cpu().numpy(),
                'pred_labels': pred_dict['pred_labels'].cpu().numpy(),
                'num_detections': len(pred_dict['pred_boxes'])
            }
            all_predictions.append(frame_result)
    
    logger.info('Inference completed.')
    
    # 根据return_all_frames决定返回值
    if len(all_predictions) == 1 and not return_all_frames:
        return all_predictions[0]
    else:
        return all_predictions


def format_predictions(pred_dict: Dict) -> str:
    """
    格式化预测结果，便于打印
    
    Args:
        pred_dict: 预测结果字典
    
    Returns:
        格式化字符串
    """
    boxes = pred_dict['pred_boxes']
    scores = pred_dict['pred_scores']
    labels = pred_dict['pred_labels']
    
    if len(boxes) == 0:
        return "No detections"
    
    lines = [f"Detected {len(boxes)} objects:"]
    for i in range(len(boxes)):
        lines.append(
            f"  {i+1:2d}. Label: {labels[i]:2d} | "
            f"Score: {scores[i]:.3f} | "
            f"Box: [{boxes[i,0]:.2f}, {boxes[i,1]:.2f}, {boxes[i,2]:.2f}, "
            f"{boxes[i,3]:.2f}, {boxes[i,4]:.2f}, {boxes[i,5]:.2f}, {boxes[i,6]:.2f}]"
        )
    
    return '\n'.join(lines)


# ==================== 使用示例 ====================

def example_usage():
    """使用示例函数"""
    
    # 示例1：推理单个文件
    result = openpcdet_inference(
        data_path='path/to/your/pointcloud.bin',
        config_path='cfgs/kitti_models/pointpillar.yaml',
        ckpt_path='checkpoints/pointpillar_7728.pth',
        ext='.bin'
    )
    
    # 打印结果
    print("Single file result:")
    print(format_predictions(result))
    
    # 示例2：推理整个目录
    results = openpcdet_inference(
        data_path='path/to/pointclouds/',
        config_path='cfgs/kitti_models/pointpillar.yaml',
        ckpt_path='checkpoints/pointpillar_7728.pth',
        ext='.bin',
        return_all_frames=True
    )
    
    # 打印所有帧的结果
    for frame_result in results:
        print(f"\nFrame {frame_result['frame_id']}:")
        print(format_predictions(frame_result))
    
    # 示例3：手动加载模型，然后逐帧推理（适合流式处理）
    model, cfg, _ = load_model(
        config_path='cfgs/kitti_models/pointpillar.yaml',
        ckpt_path='checkpoints/pointpillar_7728.pth'
    )
    
    # 模拟实时点云流
    points_list = [np.random.randn(10000, 4).astype(np.float32) for _ in range(5)]
    
    for i, points in enumerate(points_list):
        result = infer_single_frame(
            model=model,
            points=points,
            cfg=cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist']
        )
        print(f"Frame {i}: {len(result['pred_boxes'])} detections")


if __name__ == '__main__':
    # 可以直接调用函数，传入参数
    import sys
    
    # 如果作为脚本运行，可以从命令行参数获取输入
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        config_path = sys.argv[2] if len(sys.argv) > 2 else 'cfgs/kitti_models/pointpillar.yaml'
        ckpt_path = sys.argv[3] if len(sys.argv) > 3 else 'checkpoints/pointpillar_7728.pth'
        
        result = openpcdet_inference(
            data_path=data_path,
            config_path=config_path,
            ckpt_path=ckpt_path
        )
        
        print(format_predictions(result))
    else:
        # 如果没有命令行参数，运行示例
        print("Please provide data_path, config_path, and ckpt_path as arguments")
        print("Example: python script.py /path/to/data.bin config.yaml model.pth")
