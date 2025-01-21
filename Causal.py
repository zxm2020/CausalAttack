

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
import numpy as np
from torch import nn
import torch.optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 使用预训练的ResNet18作为特征提取器
        self.model = nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-1])  # 移除最后的全连接层

    def forward(self, x):
        with torch.no_grad():  # 特征提取时不需要梯度
            features = self.model(x)
        return features.view(features.size(0), -1)  # 展平


# 定义高斯核函数
def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算高斯核矩阵
    x: tensor of shape (batch_size, feature_dim)
    y: tensor of shape (batch_size, feature_dim)
    """
    n_samples = x.size(0) + y.size(0)
    total = torch.cat([x, y], dim=0)  # (2*batch_size, feature_dim)

    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))

    L2_distance = ((total0 - total1) ** 2).sum(2)  # (2*batch_size, 2*batch_size)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)

    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]

    return sum(kernel_val)  # (2*batch_size, 2*batch_size)


# 定义MMD损失函数
def mmd_loss(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算两组特征之间的MMD
    x: tensor of shape (batch_size, feature_dim)
    y: tensor of shape (batch_size, feature_dim)
    """
    batch_size = x.size(0)

    kernels = gaussian_kernel(x, y, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    loss = torch.mean(XX + YY - XY - YX)
    return loss


# 定义Causal Loss函数
def causal_loss(feature_extractor, x_ref, x_adv, device='cuda'):
    """
    计算因果损失
    feature_extractor: 特征提取器模型
    x_ref: 参考图像，tensor形状 (batch_size, 3, H, W)
    x_adv: 对抗图像，tensor形状 (batch_size, 3, H, W)
    device: 计算设备
    """
    x_ref = x_ref.to(device)
    x_adv = x_adv.to(device)

    # 提取特征
    S_P1 = feature_extractor(x_ref)  # (batch_size, feature_dim)
    S_P2 = feature_extractor(x_adv)  # (batch_size, feature_dim)

    # 计算MMD损失
    loss = mmd_loss(S_P1, S_P2)

    return loss


def add_black_square(image):
    # 图像的宽度和高度
    _,_,height, width= image.shape
    # 黑色方块的尺寸
    square_width, square_height = 240, 320
    # 计算黑色方块的起始坐标
    start_x = (width - square_width) // 2
    start_y = (height - square_height) // 2
    black_square = np.zeros((1, 3, square_height, square_width))
    # 在图像上添加黑色方块
    image[:,:,start_y:start_y+square_height, start_x:start_x+square_width] = black_square
    return image
def get_dimensions(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_dimensions(lst[0]) if lst else []
    else:
        return []

def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescales (xyxy) bounding boxes from img1_shape to img0_shape, optionally using provided `ratio_pad`."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
def lab_loss(perturbed_image_lab):
    # 从 LAB 图像中分离 L, A, B 通道
    L, A, B = perturbed_image_lab[:, 0, :, :], perturbed_image_lab[:, 1, :, :], perturbed_image_lab[:, 2, :, :]

    # A 和 B 的范围限制：确保它们不会太大或太小，可以使用平滑 L1 损失
    # 目标是让 A 和 B 的值保持在适当的范围内，可以设定一个目标范围 [min_val, max_val]
    min_val = -100  # A 和 B 通道的最小值
    max_val = 100  # A 和 B 通道的最大值

    # 计算 A 和 B 的约束损失
    loss_a = torch.mean(F.relu(A - min_val)) + torch.mean(F.relu(max_val - A))  # A 要在 [min_val, max_val] 范围内
    loss_b = torch.mean(F.relu(B - min_val)) + torch.mean(F.relu(max_val - B))  # B 要在 [min_val, max_val] 范围内

    # 总损失是 A 和 B 损失的加权和
    total_loss = loss_a + loss_b
    return total_loss


def blend_images_except_black(image_a, image_b):
    # 打开图片A和图片B，并确保它们是RGBA模式
    # image_a = Image.open(image_a_path).convert("RGBA")
    # image_b = Image.open(image_b_path).convert("RGBA")

    # 确保图片A和图片B的尺寸一致
    if image_a.size != image_b.size:
        image_a = image_a.resize(image_b.size)

    # 创建一个新的图像用于输出
    result = Image.new("RGBA", image_b.size)

    # 遍历图B的每个像素
    for x in range(image_b.width):
        for y in range(image_b.height):
            # 获取图B的像素
            r_b, g_b, b_b, a_b = image_b.getpixel((x, y))

            # 如果是纯黑色的像素，直接保留图B的像素值
            if r_b == 81 and g_b == 73 and b_b == 61:
            # if 71<r_b < 91 and 63<r_b < 83 and 51<r_b < 71:
                result.putpixel((x, y), (r_b, g_b, b_b, a_b))
                continue

            # 计算图B像素的亮度（白色程度）
            brightness = (r_b + g_b + b_b) / (3 * 255)  # 范围[0, 1]

            # 获取图A的像素
            r_a, g_a, b_a, a_a = image_a.getpixel((x, y))

            # 计算叠加后的透明度（由图A的透明度和图
            # 计算叠加后的透明度（由图A的透明度和图B的亮度共同决定）
            blended_alpha = int(a_a * brightness)  # 根据亮度调整透明度

            # 计算叠加后的RGB值（简单的线性混合公式）
            new_r = int((r_a * blended_alpha + r_b * (255 - blended_alpha)) / 255)
            new_g = int((g_a * blended_alpha + g_b * (255 - blended_alpha)) / 255)
            new_b = int((b_a * blended_alpha + b_b * (255 - blended_alpha)) / 255)

            # 创建新的像素
            new_pixel = (new_r, new_g, new_b, 255)  # 结果图像保持完全不透明

            # 放置到结果图像
            result.putpixel((x, y), new_pixel)
            return result


def rgb_to_lab_tensor(image):
    """
    将输入的RGB图像tensor转换为LAB图像tensor
    该函数不涉及numpy操作，确保梯度可以传播。
    """
    # 确保图像在[0, 1]范围内 (假设输入图像是[0, 255]范围内的整数)
    image = image # 如果是 [0, 255] 范围的图像，先进行归一化

    # 线性RGB到XYZ转换矩阵
    def rgb_to_xyz(rgb):
        # 线性RGB转换到XYZ的系数 (D65标准光源)
        M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float32)

        # 确保rgb也是float32类型，避免dtype不匹配
        rgb = rgb.to(torch.float32)  # 将rgb转换为float32类型

        return torch.matmul(rgb, M.T)/255

    # XYZ到LAB转换
    def xyz_to_lab(xyz):
        # 假设D65光源 (标准白点)
        epsilon = 0.008856
        kappa = 903.3

        # 对RGB进行非线性化变换
        xyz = xyz / torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)

        # 调整XYZ值
        f = torch.where(xyz > epsilon, torch.pow(xyz, 1/3), (kappa * xyz + 16) / 116)

        # 计算L, a, b
        L = (116 * f[:, 1]) - 16
        a = 500 * (f[:, 0] - f[:, 1])
        b = 200 * (f[:, 1] - f[:, 2])

        return torch.stack([L, a, b], dim=-1)

    # 对于每个像素，RGB到LAB转换
    batch_size, channels, height, width = image.shape
    lab_image = torch.zeros_like(image)

    for i in range(batch_size):
        for j in range(channels):
            rgb = image[i, j, :, :].view(-1, 3)  # 获取单个通道
            xyz = rgb_to_xyz(rgb)  # 将RGB转换为XYZ
            lab = xyz_to_lab(xyz)  # 将XYZ转换为LAB

            # 填充到输出张量
            lab_image[i, j, :, :] = lab.view(height, width)

    return lab_image
# @smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.5,  # confidence threshold
    iou_thres=0.5,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    """
    Runs YOLOv5 detection inference on various sources like images, videos, directories, streams, etc.

    Args:
        weights (str | Path): Path to the model weights file or a Triton URL. Default is 'yolov5s.pt'.
        source (str | Path): Input source, which can be a file, directory, URL, glob pattern, screen capture, or webcam
            index. Default is 'data/images'.
        data (str | Path): Path to the dataset YAML file. Default is 'data/coco128.yaml'.
        imgsz (tuple[int, int]): Inference image size as a tuple (height, width). Default is (640, 640).
        conf_thres (float): Confidence threshold for detections. Default is 0.25.
        iou_thres (float): Intersection Over Union (IOU) threshold for non-max suppression. Default is 0.45.
        max_det (int): Maximum number of detections per image. Default is 1000.
        device (str): CUDA device identifier (e.g., '0' or '0,1,2,3') or 'cpu'. Default is an empty string, which uses the
            best available device.
        view_img (bool): If True, display inference results using OpenCV. Default is False.
        save_txt (bool): If True, save results in a text file. Default is False.
        save_csv (bool): If True, save results in a CSV file. Default is False.
        save_conf (bool): If True, include confidence scores in the saved results. Default is False.
        save_crop (bool): If True, save cropped prediction boxes. Default is False.
        nosave (bool): If True, do not save inference images or videos. Default is False.
        classes (list[int]): List of class indices to filter detections by. Default is None.
        agnostic_nms (bool): If True, perform class-agnostic non-max suppression. Default is False.
        augment (bool): If True, use augmented inference. Default is False.
        visualize (bool): If True, visualize feature maps. Default is False.
        update (bool): If True, update all models' weights. Default is False.
        project (str | Path): Directory to save results. Default is 'runs/detect'.
        name (str): Name of the current experiment; used to create a subdirectory within 'project'. Default is 'exp'.
        exist_ok (bool): If True, existing directories with the same name are reused instead of being incremented. Default is
            False.
        line_thickness (int): Thickness of bounding box lines in pixels. Default is 3.
        hide_labels (bool): If True, do not display labels on bounding boxes. Default is False.
        hide_conf (bool): If True, do not display confidence scores on bounding boxes. Default is False.
        half (bool): If True, use FP16 half-precision inference. Default is False.
        dnn (bool): If True, use OpenCV DNN backend for ONNX inference. Default is False.
        vid_stride (int): Stride for processing video frames, to skip frames between processing. Default is 1.

    Returns:
        None

    Examples:
        ```python
        from ultralytics import run

        # Run inference on an image
        run(source='data/images/example.jpg', weights='yolov5s.pt', device='0')

        # Run inference on a video with specific confidence threshold
        run(source='data/videos/example.mp4', weights='yolov5s.pt', conf_thres=0.4, device='0')
        ```
    """
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path_dig, vid_writer_dig = [None] * bs, [None] * bs
    vid_path_phy, vid_writer_phy = [None] * bs, [None] * bs
    vid_path_pro, vid_writer_pro = [None] * bs, [None] * bs
    # vid_path_dig_vedio, vid_writer_dig_vedio = [None] * bs, [None] * bs
    # vid_path_phy_vedio, vid_writer_phy_vedio = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    # perturbation = np.random.uniform(-20, 20, (1, 3, 480, 640))


    learning_rate = 0.1
    import matplotlib.pyplot as plt
    import time
    from PIL import Image

    # 初始化 VideoWriter，准备保存视频
    fps = 10      # 帧率 (例如 30 帧每秒)
    frame_height, frame_width = 480, 640  # 图像的高度和宽度

    # 使用 mp4 格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 格式
    phy_no_detect = cv2.VideoWriter(str(Path(str(save_dir / "phy_no_detect" )).with_suffix(".mp4")), fourcc, fps, (frame_width, frame_height))
    dig_no_detect = cv2.VideoWriter(str(Path(str(save_dir / "dig_no_detect" )).with_suffix(".mp4")), fourcc, fps, (frame_width, frame_height))
    boxes_before = []
    perturbation_grad_before = torch.zeros((1, 3, 480, 640))
    for path, im, im0s, vid_cap, s in dataset:
        # im(1, 3, 480, 640)
        # im0s(1, 480, 640, 3)
        camera_im = im
        # 从 camera_im 提取出图像，调整形状为 (480, 640, 3)
        camera_im_frame = np.transpose(camera_im.squeeze(), (1, 2, 0))  # (1, 3, 480, 640) -> (480, 640, 3)

        # 确保图像是 uint8 类型
        camera_im_frame = np.clip(camera_im_frame[:, :, ::-1], 0, 255).astype(np.uint8)

        # 将该帧写入视频
        phy_no_detect.write(camera_im_frame)
        perturbation = np.zeros((1, 3, 480, 640))
        # perturbation = np.zeros(im.shape)
        # perturbation = np.random.rand(1,3,480,640)
        for epoch in range(5):
            perturbation = torch.tensor(perturbation, requires_grad=True)
            # print("1",perturbation)
            perturbation_lab = rgb_to_lab_tensor(perturbation)
            # perturbed_image = torch.tensor(im, dtype=torch.float32).to(perturbation.device) + perturbation
            perturbed_image = blend_images_except_black(im,perturbation)
            # print("2", perturbed_image)
            im_ori = torch.clamp(perturbed_image, 0, 255)
            im_transposed = np.transpose(im_ori.detach().numpy(), (0, 2, 3, 1))
            im_transformed = im_transposed[:, :, :, [2, 1, 0]]
            im_transformed = im_transposed[:, :, :, [2, 1, 0]]
            im0s_dig = im_transformed

            with dt[0]:
                im_ori = im_ori.to(model.device)
                im_ori = im_ori.half() if model.fp16 else im_ori.float()  # uint8 to fp16/32


                im_ori /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im_ori.shape) == 3:
                    im_ori = im_ori[None]  # expand for batch dim
                if model.xml and im_ori.shape[0] > 1:
                    ims = torch.chunk(im_ori, im_ori.shape[0], 0)


            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                if model.xml and im_ori.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                        else:
                            pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)

                    pred = [pred, None]
                else:
                    pred = model(im_ori, augment=augment, visualize=visualize)
                    # print("3", pred)
                    # grad_fn = < ClampBackward1 >
                    #  grad_fn = < DivBackward0 >
            # truck12
            # person5
            flag_person = ((pred[0][0][:, 12] > 0.4) & (pred[0][0][:, 4] > 0.2)) * 1
            loss_yolo_person = ((flag_person * pred[0][0][:, 4])).sum()

            flag_others = ((pred[0][0][:, 12] < 0.4) & (pred[0][0][:, 4] > 0.8)) * 1
            loss_yolo_others = (flag_others * pred[0][0][:, 4]).sum()
            img_path_adv = 'path_to_adversarial_image.jpg'  # 替换为对抗图像路径
            img_ref = Image.open(img_path_ref).convert('RGB')
            # 预处理图像
            img_ref_tensor = preprocess(img_ref).unsqueeze(0)  # 添加batch维度
            img_adv_tensor = preprocess(im_ori).unsqueeze(0)  # 添加batch维度
            c_loss = causal_loss(feature_extractor, img_ref_tensor, img_adv_tensor, device)
            loss_lab = custom_loss(perturbation_lab)
            loss = -loss_yolo_person + loss_yolo_others +c_loss+loss_lab
            loss.requires_grad_(True)

            # 确保在进行反向传播之前清除之前的梯度
            perturbation.grad = None
            # perturbation.grad.zero_()
            # 反向传播
            loss.backward()
            perturbation_grad_all = perturbation.grad.data
            perturbation_grad = torch.ones((1, 3, 480, 640))
            # perturbation_grad = torch.rand(1,3,480,640)

            # 获取分类置信度部分
            confidences = pred[0][0, :, -80:]
            confidences = confidences.clone().cpu().detach().numpy()
            pred_max = pred[0].cpu().detach().numpy()
            # 找到最大置信度的索引
            # print(confidences.shape)
            max_indices = np.argmax(confidences, axis=1)

            # 创建一个布尔掩码，表示满足条件的备选框
            # mask = ((max_indices == 0) & (pred_max[0][:, 4] > 0.5))
            mask = ((max_indices == 7) & (pred_max[0][:, 4] > 0.5))


            # 使用布尔掩码提取满足条件的box
            boxes = pred_max[0][mask, :4]
            if len(boxes) > 0:
                boxes_before = boxes
                for i in range(len(boxes_before)):
                    box = xywh2xyxy(boxes_before[i])

                    perturbation_grad[:, :, int(box[1]):int(box[3]),
                    int(box[0]):int(box[2])] = perturbation_grad_all[:, :, int(box[1]):int(box[3]),
                                               int(box[0]):int(box[2])]
                    perturbation_grad_before = perturbation_grad
            else:
                boxes_before = boxes_before
                perturbation_grad = perturbation_grad_before


            perturbation = perturbation + learning_rate * perturbation_grad.sign()
            # perturbation = perturbation + learning_rate * perturbation_grad_all.sign()

        dig_perturbed_image = perturbed_image
        # 从 camera_im 提取出图像，调整形状为 (480, 640, 3)
        dig_im_frame = np.transpose(dig_perturbed_image.detach().numpy().squeeze(), (1, 2, 0))  # (1, 3, 480, 640) -> (480, 640, 3)

        # 确保图像是 uint8 类型
        dig_im_frame = np.clip(dig_im_frame[:, :, ::-1], 0, 255).astype(np.uint8)

        # 将该帧写入视频
        dig_no_detect.write(dig_im_frame)
        with dt[2]:
            camera_im = torch.from_numpy(camera_im).to(model.device)
            camera_im = camera_im.half() if model.fp16 else camera_im.float()  # uint8 to fp16/32
            camera_im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(camera_im.shape) == 3:
                camera_im = camera_im[None]  # expand for batch dim
            if model.xml and camera_im.shape[0] > 1:
                ims = torch.chunk(camera_im, camera_im.shape[0], 0)
            pred_camera = model(camera_im, augment=augment, visualize=visualize)
            conf_thres = 0.5
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            pred_camera = non_max_suppression(pred_camera, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)



        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        # for i, det in enumerate(pred):  # per image
        i = 0
        det = pred[0]
        det_camera = pred_camera[0]
        seen += 1
        if webcam:  # batch_size >= 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            im0 = np.transpose(im_ori.cpu().detach().numpy(), (0, 2, 3, 1))[0] * 255
            im0_camera = im0s[i][:, :, [2, 1, 0]]
            # im0shape = np.array(im0)
            # print(im0shape.shape)(480, 640, 3)
            s += f"{i}: "
        else:
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
        im0 = np.array(im0)
        im0_camera = np.array(im0_camera)
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        txt_path = str(save_dir / "labels" / p.stem) + (
            "" if dataset.mode == "image" else f"_{frame}")  # im.txt
        s += "%gx%g " % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        # # 将 im0 转换为连续张量
        # im0 = im0.contiguous()
        # 将图像转换为连续的数组
        im0 = np.ascontiguousarray(im0)
        im0_camera = np.ascontiguousarray(im0_camera)
        # dig_vedio = im0
        # phy_vedio = im0_camera
        # print("dig_vedio",dig_vedio.shape)

        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        annotator_camera = Annotator(im0_camera, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = names[c] if hide_conf else f"{names[c]}"
                confidence = float(conf)
                confidence_str = f"{confidence:.2f}"

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    annotator.box_label(xyxy, label, color=colors(c, True))
        if len(det_camera):
            # Rescale boxes from img_size to im0 size
            det_camera[:, :4] = scale_boxes(im.shape[2:], det_camera[:, :4], im0_camera.shape).round()

            # Print results
            for c in det_camera[:, 5].unique():
                n = (det_camera[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy_camera, conf_camera, cls_camera in reversed(det_camera):
                c_camera = int(cls_camera)  # integer class
                label_camera = names[c_camera] if hide_conf else f"{names[c_camera]}"
                confidence_camera = float(conf_camera)
                confidence_camera = f"{confidence_camera:.2f}"

                if save_img or save_crop or view_img:  # Add bbox to image
                    c_camera = int(cls_camera)  # integer class
                    label_camera = None if hide_labels else (names[c_camera] if hide_conf else f"{names[c_camera]} {conf_camera:.2f}")
                    annotator_camera.box_label(xyxy_camera, label_camera, color=colors(c_camera, True))

        # Stream results
        im0 = annotator.result()
        im0_camera = annotator_camera.result()
        # dig_vedio = dig_vedio.result()
        # phy_vedio = phy_vedio.result()
        im0 = cv2.cvtColor(np.array(im0), cv2.COLOR_RGB2BGR)
        im0_camera = cv2.cvtColor(np.array(im0_camera), cv2.COLOR_RGB2BGR)
        # dig_vedio = cv2.cvtColor(np.array(dig_vedio), cv2.COLOR_RGB2BGR)
        # phy_vedio = cv2.cvtColor(np.array(phy_vedio), cv2.COLOR_RGB2BGR)
        im0 = cv2.convertScaleAbs(im0)
        im0_camera = cv2.convertScaleAbs(im0_camera)
        # dig_vedio = cv2.convertScaleAbs(dig_vedio)
        # phy_vedio = cv2.convertScaleAbs(phy_vedio)
        perturbation_output = np.transpose(perturbation.cpu().detach().numpy(), (0, 2, 3, 1))[0]
        perturbation_output = perturbation_output[:, :, ::-1]
        # perturbation_output = cv2.cvtColor(np.array(perturbation_output), cv2.COLOR_RGB2BGR)
        # perturbation_output = cv2.convertScaleAbs(perturbation_output)
        # image = Image.fromarray((im0).astype(np.uint8))
        # image.save('images/output_image.jpg')

        if view_img:
            # if platform.system() == "Linux" and p not in windows:
            #     windows.append(p)
            #     cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #     cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

            height, width = perturbation_output.shape[:2]

            # 计算新的宽度和高度，以保持图像的纵横比
            new_width = int(width * 1.5)  # 例如，将宽度增加50%
            new_height = int(height * 1.5)  # 同样地，将高度增加50%

            # 调整图像大小以适应新的宽度和高度
            resized_image = cv2.resize(perturbation_output, (new_width, new_height))

            # 创建一个名为 "Output Image" 的窗口，并允许调整大小
            cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)

            # 显示调整后的图像
            cv2.imshow("Output Image", resized_image)

            cv2.imshow("dig", im0)
            cv2.imshow("phy", im0_camera)
            cv2.waitKey(1)  # 1 millisecond
        # Save results (image with detections)
        if save_img:
            if dataset.mode == "image":
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path_dig[i] != save_path:  # new video
                    vid_path_dig[i] = save_path
                    if isinstance(vid_writer_dig[i], cv2.VideoWriter):
                        vid_writer_dig[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 10, im0.shape[1], im0.shape[0]
                    save_path_dig = str(Path(str(save_dir / "dig" )).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                    save_path_phy = str(Path(str(save_dir / "phy")).with_suffix(".mp4"))
                    save_path_pro = str(Path(str(save_dir / "pro")).with_suffix(".mp4"))
                    # save_path_dig_vedio = str(Path(str(save_dir / "dig_vedio")).with_suffix(".mp4"))
                    # save_path_phy_vedio = str(Path(str(save_dir / "phy_vedio")).with_suffix(".mp4"))
                    vid_writer_dig[i] = cv2.VideoWriter(save_path_dig, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer_phy[i] = cv2.VideoWriter(save_path_phy, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer_pro[i] = cv2.VideoWriter(save_path_pro, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    # vid_writer_dig_vedio[i] = cv2.VideoWriter(save_path_dig_vedio, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    # vid_writer_phy_vedio[i] = cv2.VideoWriter(save_path_phy_vedio, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

                im0_8u_dig = cv2.convertScaleAbs(im0/255, alpha=(255.0))
                im0_8u_phy = cv2.convertScaleAbs(im0_camera/255, alpha=(255.0))
                im0_8u_pro = cv2.convertScaleAbs(perturbation_output, alpha=(255.0))
                # im0_8u_dig_vedio = cv2.convertScaleAbs(dig_vedio/255, alpha=(255.0))
                # im0_8u_phy_vedio = cv2.convertScaleAbs(phy_vedio/255, alpha=(255.0))

                vid_writer_dig[i].write(im0_8u_dig)
                vid_writer_phy[i].write(im0_8u_phy)
                vid_writer_pro[i].write(im0_8u_pro)
                # vid_writer_dig_vedio[i].write(im0_8u_dig_vedio)
                # vid_writer_phy_vedio[i].write(im0_8u_phy_vedio)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    phy_no_detect.release()
    dig_no_detect.release()
    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """
    Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

    Args:
        --weights (str | list[str], optional): Model path or Triton URL. Defaults to ROOT / 'yolov5s.pt'.
        --source (str, optional): File/dir/URL/glob/screen/0(webcam). Defaults to ROOT / 'data/images'.
        --data (str, optional): Dataset YAML path. Provides dataset configuration information.
        --imgsz (list[int], optional): Inference size (height, width). Defaults to [640].
        --conf-thres (float, optional): Confidence threshold. Defaults to 0.25.
        --iou-thres (float, optional): NMS IoU threshold. Defaults to 0.45.
        --max-det (int, optional): Maximum number of detections per image. Defaults to 1000.
        --device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        --view-img (bool, optional): Flag to display results. Defaults to False.
        --save-txt (bool, optional): Flag to save results to *.txt files. Defaults to False.
        --save-csv (bool, optional): Flag to save results in CSV format. Defaults to False.
        --save-conf (bool, optional): Flag to save confidences in labels saved via --save-txt. Defaults to False.
        --save-crop (bool, optional): Flag to save cropped prediction boxes. Defaults to False.
        --nosave (bool, optional): Flag to prevent saving images/videos. Defaults to False.
        --classes (list[int], optional): List of classes to filter results by, e.g., '--classes 0 2 3'. Defaults to None.
        --agnostic-nms (bool, optional): Flag for class-agnostic NMS. Defaults to False.
        --augment (bool, optional): Flag for augmented inference. Defaults to False.
        --visualize (bool, optional): Flag for visualizing features. Defaults to False.
        --update (bool, optional): Flag to update all models in the model directory. Defaults to False.
        --project (str, optional): Directory to save results. Defaults to ROOT / 'runs/detect'.
        --name (str, optional): Sub-directory name for saving results within --project. Defaults to 'exp'.
        --exist-ok (bool, optional): Flag to allow overwriting if the project/name already exists. Defaults to False.
        --line-thickness (int, optional): Thickness (in pixels) of bounding boxes. Defaults to 3.
        --hide-labels (bool, optional): Flag to hide labels in the output. Defaults to False.
        --hide-conf (bool, optional): Flag to hide confidences in the output. Defaults to False.
        --half (bool, optional): Flag to use FP16 half-precision inference. Defaults to False.
        --dnn (bool, optional): Flag to use OpenCV DNN for ONNX inference. Defaults to False.
        --vid-stride (int, optional): Video frame-rate stride, determining the number of frames to skip in between
            consecutive frames. Defaults to 1.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Example:
        ```python
        from ultralytics import YOLOv5
        args = YOLOv5.parse_opt()
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov3.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "4", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 model inference based on provided command-line arguments, validating dependencies before running.

    Args:
        opt (argparse.Namespace): Command-line arguments for YOLOv5 detection. See function `parse_opt` for details.

    Returns:
        None

    Note:
        This function performs essential pre-execution checks and initiates the YOLOv5 detection process based on user-specified
        options. Refer to the usage guide and examples for more information about different sources and formats at:
        https://github.com/ultralytics/ultralytics

    Example usage:

    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
    ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
