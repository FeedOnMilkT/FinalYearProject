"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
# open3d.visualization.webrtc_server.enable_webrtc()
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):

    open3d.visualization.gui.Application.instance.initialize()    

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    render_option = vis.get_render_option()
    # vis.create_window()
    # vis.get_render_option().point_size = 1.0
    # vis.get_render_option().background_color = np.zeros(3)

    try:
        vis.create_window(visible=True)
        render_option.point_size = 1.0
        render_option.background_color = np.zeros(3)
    except Exception as e:
        print(f"Warning: Failed to create visible window, falling back to offscreen rendering: {e}")
        # 设置环境变量以使用软件渲染
        import os
        os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        
        # 重新尝试创建窗口
        vis.create_window(visible=True, width=1920, height=1080)
        render_option.point_size = 1.0
        render_option.background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()

def draw_scenes_with_webrtc(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    # Create list to hold geometries
    geometries = []
    
    if draw_origin:
        # Create coordinate frame
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        geometries.append(axis_pcd)

    # Create point cloud
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    # Add point cloud to geometries
    geometries.append(pts)

    if gt_boxes is not None:
        gt_box_geometries = draw_box_with_webrtc(gt_boxes, (0, 0, 1))
        geometries.extend(gt_box_geometries)

    if ref_boxes is not None:
        ref_box_geometries = draw_box_with_webrtc(ref_boxes, (0, 1, 0), ref_labels, ref_scores)
        geometries.extend(ref_box_geometries)

    # Use WebRTC to visualize
    # open3d.visualization.webrtc_server.enable_webrtc()
    open3d.visualization.draw(
        geometries, 
        show_ui=True,
        point_size=1,
        # background_color=(0, 0, 0),
        width=1920,
        height=1080)


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

def draw_box_with_webrtc(boxes, color=(0, 1, 0), ref_labels=None, score=None):

    box_geometries = []

    for i in range(boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        # else:
            # line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        box_geometries.append(line_set)

    return box_geometries

