# 文件：isw_setup/isw_basic.py
from veros import VerosSetup, veros_method
import numpy as np

class InternalWaveSetup(VerosSetup):
    """Idealized internal wave propagation in a 2D stratified fluid"""

    @veros_method
    def set_grid(self):
        # 模拟 100 km 长、1 km 深的剖面
        self.x, self.dx = np.linspace(0, 100e3, 256, retstep=True)
        self.z, self.dz = np.linspace(-1000, 0, 128, retstep=True)

    @veros_method
    def set_initial_conditions(self):
        # 初始温度场 -> 稳定分层 + 一个局部扰动
        z = self.z
        temp_background = 15 + 10 * np.tanh((z + 500) / 100)
        self.state.temp[...] = temp_background[np.newaxis, :]

        # 添加一个内部波扰动（例如 Gaussian 模态）
        A = 0.1  # 振幅 (°C)
        x0 = 30e3
        sigma = 5e3
        self.state.temp[...] += A * np.exp(-((self.x[:, np.newaxis] - x0) ** 2) / (2 * sigma**2)) * np.sin(np.pi * (z + 1000) / 1000)

    @veros_method
    def set_forcing(self):
        pass  # 可后续添加风应力或边界流入扰动

if __name__ == "__main__":
    sim = InternalWaveSetup()
    sim.run()
