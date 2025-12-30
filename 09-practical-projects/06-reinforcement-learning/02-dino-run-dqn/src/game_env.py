"""
Chrome Dino游戏环境

提供两种模式：
1. 浏览器模式：通过Selenium控制真实的Chrome浏览器
2. 模拟模式：用于测试的简化环境
"""

import numpy as np
import time

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from PIL import Image
    import io
    import base64
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class DinoGameEnv:
    """
    Chrome Dino游戏环境（浏览器模式）
    
    通过Selenium控制Chrome浏览器，与真实游戏交互。
    
    动作空间：
        0: 不操作
        1: 跳跃
    """
    
    GAME_URL = "chrome://dino"
    INIT_SCRIPT = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
    GET_CANVAS_SCRIPT = "return document.getElementById('runner-canvas').toDataURL().substring(22)"
    
    def __init__(self, chrome_driver_path=None):
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is required. Install with: pip install selenium")
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required. Install with: pip install Pillow")
        
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        
        if chrome_driver_path:
            self.driver = webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)
        else:
            self.driver = webdriver.Chrome(options=chrome_options)
        
        self.driver.set_window_position(x=0, y=0)
        self.driver.get(self.GAME_URL)
        self.driver.execute_script("Runner.config.ACCELERATION=0")
        self.driver.execute_script(self.INIT_SCRIPT)
    
    def reset(self):
        """重置游戏"""
        self.driver.execute_script("Runner.instance_.restart()")
        time.sleep(0.1)
        self._press_up()
        return self._get_screen()
    
    def step(self, action):
        """执行动作"""
        if action == 1:
            self._press_up()
        
        time.sleep(0.05)
        
        screen = self._get_screen()
        crashed = self._is_crashed()
        score = self._get_score()
        
        if crashed:
            reward = -1.0
            done = True
        else:
            reward = 0.1
            done = False
        
        return screen, reward, done, {"score": score}
    
    def _press_up(self):
        """按下跳跃键"""
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)
    
    def _is_crashed(self):
        """检查是否碰撞"""
        return self.driver.execute_script("return Runner.instance_.crashed")
    
    def _get_score(self):
        """获取当前分数"""
        score_array = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
        return int(''.join(score_array)) if score_array else 0
    
    def _get_screen(self):
        """获取游戏画面"""
        image_b64 = self.driver.execute_script(self.GET_CANVAS_SCRIPT)
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        return np.array(image)
    
    def close(self):
        """关闭浏览器"""
        self.driver.quit()


class DinoGameSimulator:
    """
    Dino游戏模拟器（用于测试）
    
    简化的游戏逻辑，不需要浏览器，用于快速测试代码。
    """
    
    def __init__(self, screen_size=(150, 600)):
        self.screen_height, self.screen_width = screen_size
        self.reset()
    
    def reset(self):
        """重置游戏状态"""
        self.dino_y = 0
        self.dino_velocity = 0
        self.obstacles = []
        self.score = 0
        self.game_speed = 5
        self.is_jumping = False
        self._spawn_obstacle()
        return self._render()
    
    def step(self, action):
        """执行一步"""
        if action == 1 and not self.is_jumping:
            self.dino_velocity = 15
            self.is_jumping = True
        
        self.dino_y += self.dino_velocity
        self.dino_velocity -= 1
        
        if self.dino_y <= 0:
            self.dino_y = 0
            self.is_jumping = False
        
        for obs in self.obstacles:
            obs['x'] -= self.game_speed
        
        self.obstacles = [obs for obs in self.obstacles if obs['x'] > -50]
        
        if len(self.obstacles) == 0 or self.obstacles[-1]['x'] < self.screen_width - 200:
            if np.random.random() < 0.02:
                self._spawn_obstacle()
        
        done = False
        reward = 0.1
        
        for obs in self.obstacles:
            if 40 < obs['x'] < 80 and self.dino_y < obs['height']:
                done = True
                reward = -1.0
                break
            if obs['x'] < 40 and not obs.get('passed', False):
                obs['passed'] = True
                self.score += 1
                reward = 1.0
        
        return self._render(), reward, done, {"score": self.score}
    
    def _spawn_obstacle(self):
        """生成障碍物"""
        height = np.random.randint(20, 50)
        self.obstacles.append({
            'x': self.screen_width,
            'height': height,
            'passed': False
        })
    
    def _render(self):
        """渲染游戏画面"""
        screen = np.ones((self.screen_height, self.screen_width), dtype=np.uint8) * 255
        
        dino_x, dino_bottom = 50, self.screen_height - 30 - int(self.dino_y)
        screen[max(0, dino_bottom-40):dino_bottom, dino_x:dino_x+30] = 100
        
        for obs in self.obstacles:
            if 0 < obs['x'] < self.screen_width:
                obs_x = int(obs['x'])
                obs_bottom = self.screen_height - 30
                obs_top = obs_bottom - obs['height']
                screen[obs_top:obs_bottom, obs_x:min(obs_x+20, self.screen_width)] = 50
        
        screen[self.screen_height-30:self.screen_height-28, :] = 0
        
        return screen
    
    def close(self):
        """关闭环境"""
        pass
