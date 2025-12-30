"""
Flappy Bird游戏环境

基于Pygame实现的Flappy Bird游戏，封装为强化学习环境接口。
提供标准的step/reset接口，方便与各种RL算法对接。
"""

import numpy as np
from itertools import cycle
from numpy.random import randint

try:
    import pygame
    from pygame import Rect, init, time, display
    from pygame.event import pump
    from pygame.image import load
    from pygame.surfarray import array3d, pixels_alpha
    from pygame.transform import rotate
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class FlappyBirdEnv:
    """
    Flappy Bird强化学习环境
    
    状态空间：游戏画面（288x512 RGB图像）
    动作空间：2个离散动作
        - 0: 不操作（小鸟自然下落）
        - 1: 跳跃（小鸟向上飞）
    奖励设计：
        - 存活：+0.1
        - 通过管道：+1.0
        - 碰撞死亡：-1.0
    """
    
    def __init__(self, assets_path='assets'):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required. Install with: pip install pygame")
        
        init()
        self.fps_clock = time.Clock()
        self.screen_width = 288
        self.screen_height = 512
        self.screen = display.set_mode((self.screen_width, self.screen_height))
        display.set_caption('Flappy Bird DQN')
        
        # 加载游戏资源
        self.base_image = load(f'{assets_path}/sprites/base.png').convert_alpha()
        self.background_image = load(f'{assets_path}/sprites/background-black.png').convert()
        self.pipe_images = [
            rotate(load(f'{assets_path}/sprites/pipe-green.png').convert_alpha(), 180),
            load(f'{assets_path}/sprites/pipe-green.png').convert_alpha()
        ]
        self.bird_images = [
            load(f'{assets_path}/sprites/redbird-upflap.png').convert_alpha(),
            load(f'{assets_path}/sprites/redbird-midflap.png').convert_alpha(),
            load(f'{assets_path}/sprites/redbird-downflap.png').convert_alpha()
        ]
        
        # 碰撞检测掩码
        self.bird_hitmask = [pixels_alpha(img).astype(bool) for img in self.bird_images]
        self.pipe_hitmask = [pixels_alpha(img).astype(bool) for img in self.pipe_images]
        
        # 游戏参数
        self.fps = 30
        self.pipe_gap_size = 100
        self.pipe_velocity_x = -4
        self.min_velocity_y = -8
        self.max_velocity_y = 10
        self.downward_speed = 1
        self.upward_speed = -9
        
        self.bird_index_generator = cycle([0, 1, 2, 1])
        
        # 尺寸信息
        self.bird_width = self.bird_images[0].get_width()
        self.bird_height = self.bird_images[0].get_height()
        self.pipe_width = self.pipe_images[0].get_width()
        self.pipe_height = self.pipe_images[0].get_height()
        self.base_y = self.screen_height * 0.79
        self.base_shift = self.base_image.get_width() - self.background_image.get_width()
        
        self.reset()
    
    def reset(self):
        """重置游戏状态"""
        self.iter = 0
        self.bird_index = 0
        self.score = 0
        
        self.bird_x = int(self.screen_width / 5)
        self.bird_y = int((self.screen_height - self.bird_height) / 2)
        self.base_x = 0
        
        # 初始化管道
        self.pipes = [self._generate_pipe(), self._generate_pipe()]
        self.pipes[0]["x_upper"] = self.pipes[0]["x_lower"] = self.screen_width
        self.pipes[1]["x_upper"] = self.pipes[1]["x_lower"] = self.screen_width * 1.5
        
        self.current_velocity_y = 0
        self.is_flapped = False
        
        return self._get_frame()
    
    def _generate_pipe(self):
        """生成新管道"""
        x = self.screen_width + 10
        gap_y = randint(2, 10) * 10 + int(self.base_y / 5)
        return {
            "x_upper": x,
            "y_upper": gap_y - self.pipe_height,
            "x_lower": x,
            "y_lower": gap_y + self.pipe_gap_size
        }
    
    def _is_collided(self):
        """碰撞检测"""
        if self.bird_height + self.bird_y + 1 >= self.base_y:
            return True
        
        bird_bbox = Rect(self.bird_x, self.bird_y, self.bird_width, self.bird_height)
        
        for pipe in self.pipes:
            pipe_boxes = [
                Rect(pipe["x_upper"], pipe["y_upper"], self.pipe_width, self.pipe_height),
                Rect(pipe["x_lower"], pipe["y_lower"], self.pipe_width, self.pipe_height)
            ]
            
            if bird_bbox.collidelist(pipe_boxes) == -1:
                continue
            
            for i, pipe_box in enumerate(pipe_boxes):
                cropped = bird_bbox.clip(pipe_box)
                min_x1 = cropped.x - bird_bbox.x
                min_y1 = cropped.y - bird_bbox.y
                min_x2 = cropped.x - pipe_box.x
                min_y2 = cropped.y - pipe_box.y
                
                bird_mask = self.bird_hitmask[self.bird_index]
                pipe_mask = self.pipe_hitmask[i]
                
                if np.any(
                    bird_mask[min_x1:min_x1+cropped.width, min_y1:min_y1+cropped.height] *
                    pipe_mask[min_x2:min_x2+cropped.width, min_y2:min_y2+cropped.height]
                ):
                    return True
        
        return False
    
    def step(self, action):
        """
        执行一步动作
        
        Args:
            action: 0=不操作, 1=跳跃
        
        Returns:
            frame: 游戏画面
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        pump()
        reward = 0.1
        done = False
        
        if action == 1:
            self.current_velocity_y = self.upward_speed
            self.is_flapped = True
        
        # 更新分数
        bird_center_x = self.bird_x + self.bird_width / 2
        for pipe in self.pipes:
            pipe_center_x = pipe["x_upper"] + self.pipe_width / 2
            if pipe_center_x < bird_center_x < pipe_center_x + 5:
                self.score += 1
                reward = 1.0
                break
        
        # 更新小鸟动画
        if (self.iter + 1) % 3 == 0:
            self.bird_index = next(self.bird_index_generator)
            self.iter = 0
        self.base_x = -((-self.base_x + 100) % self.base_shift)
        
        # 更新小鸟位置
        if self.current_velocity_y < self.max_velocity_y and not self.is_flapped:
            self.current_velocity_y += self.downward_speed
        if self.is_flapped:
            self.is_flapped = False
        
        self.bird_y += min(
            self.current_velocity_y,
            self.bird_y - self.current_velocity_y - self.bird_height
        )
        if self.bird_y < 0:
            self.bird_y = 0
        
        # 更新管道位置
        for pipe in self.pipes:
            pipe["x_upper"] += self.pipe_velocity_x
            pipe["x_lower"] += self.pipe_velocity_x
        
        if 0 < self.pipes[0]["x_lower"] < 5:
            self.pipes.append(self._generate_pipe())
        if self.pipes[0]["x_lower"] < -self.pipe_width:
            del self.pipes[0]
        
        # 碰撞检测
        if self._is_collided():
            done = True
            reward = -1.0
        
        frame = self._render()
        self.iter += 1
        
        return frame, reward, done, {"score": self.score}
    
    def _render(self):
        """渲染游戏画面"""
        self.screen.blit(self.background_image, (0, 0))
        self.screen.blit(self.base_image, (self.base_x, self.base_y))
        self.screen.blit(self.bird_images[self.bird_index], (self.bird_x, self.bird_y))
        
        for pipe in self.pipes:
            self.screen.blit(self.pipe_images[0], (pipe["x_upper"], pipe["y_upper"]))
            self.screen.blit(self.pipe_images[1], (pipe["x_lower"], pipe["y_lower"]))
        
        display.update()
        self.fps_clock.tick(self.fps)
        
        return array3d(display.get_surface())
    
    def _get_frame(self):
        """获取当前画面"""
        return self._render()
    
    def close(self):
        """关闭环境"""
        pygame.quit()
