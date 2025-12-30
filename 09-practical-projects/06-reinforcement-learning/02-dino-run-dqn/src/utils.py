"""
图像处理和工具函数
"""

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def process_image(image, target_size=(80, 80)):
    """
    预处理游戏截图
    
    Args:
        image: 原始图像 (RGB或灰度)
        target_size: 目标尺寸
    
    Returns:
        处理后的图像 (H, W)，归一化到[0,1]
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required. Install with: pip install opencv-python")
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    resized = cv2.resize(gray, target_size)
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized


def stack_frames(frames, num_frames=4):
    """
    堆叠多帧图像
    
    Args:
        frames: 帧列表
        num_frames: 堆叠数量
    
    Returns:
        堆叠后的状态 (H, W, num_frames)
    """
    if len(frames) < num_frames:
        padding = [frames[0]] * (num_frames - len(frames))
        frames = padding + list(frames)
    
    return np.stack(frames[-num_frames:], axis=-1)


class FrameBuffer:
    """帧缓冲区，用于堆叠连续帧"""
    
    def __init__(self, num_frames=4, frame_size=(80, 80)):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frames = []
    
    def reset(self, initial_frame):
        """重置缓冲区"""
        processed = process_image(initial_frame, self.frame_size)
        self.frames = [processed] * self.num_frames
        return self.get_state()
    
    def add_frame(self, frame):
        """添加新帧"""
        processed = process_image(frame, self.frame_size)
        self.frames.append(processed)
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)
        return self.get_state()
    
    def get_state(self):
        """获取当前状态"""
        return np.stack(self.frames, axis=-1)
