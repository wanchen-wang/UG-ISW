#veros run isw.py

from veros import VerosSetup, veros_routine
from veros.variables import allocate, Variable
from veros.distributed import global_min, global_max
from veros.core.operators import numpy as npx, update, at

class ISWBasicSetup(VerosSetup):
    """最简内孤立波两维模型（x-z 方向，y 方向极窄）。"""

    @veros_routine
    def set_parameter(self, state):
        s = state.settings

        s.identifier = "ISW_DATA_VIRTUAL"
        s.description = "内孤立波虚拟数据生成"

        # 模型网格分辨率（可自行修改）
        total_x = 222.0  # x方向总距离（km）
        total_y = 88.8   # y方向总距离（km）
        dx = 0.2         # x方向网格间距（km）
        dy = 0.2         # y方向网格间距（km）
        s.nx = int(npx.ceil(total_x / dx))        # x 方向网格数
        s.ny = int(npx.ceil(total_x / dy))        # y 方向网格数
        s.nz = 20         # 垂向分辨率
        # 坐标（使用米制，不使用经纬度）
        s.coord_degree = False      # 使用米
        s.enable_cyclic_x = False   # x 不循环（可改为 True 得到周期性通道）
        s.x_origin = 0          # 原点东经117.5度
        s.y_origin = 0           # 原点北纬22度

        # 时间步长（内孤立波需要短程模拟）
        s.dt_mom = 30.0             # 动量方程时间步长（秒）
        s.dt_tracer = 30.0          # 示踪量时间步长（秒）
        s.runlen = 3600.0 * 6.0     # 默认模拟 6h

        # 摩擦参数设置
        s.enable_hor_friction = False
        s.enable_bottom_friction = False
        s.enable_implicit_vert_friction = True

        # 关闭水平混合
        s.enable_neutral_diffusion = False
        s.enable_skew_diffusion = False

        # 关闭湍流混合方案
        s.enable_tke = False
        s.enable_idemix = False
        s.enable_eke = False

        # 状态方程类型（TEOS-10海洋状态方程）
        s.eq_of_state_type = 3

        vs_meta = state.var_meta
        vs_meta.update(
            isw_c=Variable("isw_c", ("time",), "m/s", "内孤立波相速"),
            isw_width=Variable("isw_width", ("time",), "m", "内孤立波宽度"),
            isw_amp=Variable("isw_amp", ("time",), "m", "内孤立波波高"),
        )

    @veros_routine
    def set_grid(self, state):
        vs = state.variables
        # 设置水平网格为均匀（米）
        dx = 10.0   # x 方向 10 m 网格
        dy = 50.0   # y 方向 50 m
        vs.dxt = update(vs.dxt, at[...], dx)
        vs.dyt = update(vs.dyt, at[...], dy)

        # 垂向网格：靠近表层和密跃层更细
        import numpy as _np
        Nz = state.settings.nz
        # 拉伸网格：表层薄、底部厚
        z_edges = _np.linspace(0, 200.0, Nz + 1)  # 总深度 200 m（可调）
        dz = z_edges[1:] - z_edges[:-1]
        vs.dzt = update(vs.dzt, at[...], dz)

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        s = state.settings
        # 设置极小的科氏力，使内孤立波几乎不受旋转效应（若要完全去旋转，可设为 0）
        f0 = 0.0
        vs.coriolis_t = update(vs.coriolis_t, at[:, :], f0)

    @veros_routine
    def set_topography(self, state):
        vs = state.variables
        # 平底（kbot=0）
        vs.kbot = update(vs.kbot, at[...], 0)

    @veros_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        s = state.settings

        # 参数（可调）
        total_depth = vs.dzt.sum()               # 总深度（m）
        pyc_depth = 80.0                         # 密跃层中心深度（m）
        pyc_thickness = 8.0                      # 密跃层厚度（m）
        isw_amp = 6.0                            # 初始内孤立波位移幅度（m）
        isw_length = 200.0                       # 高斯扰动的沿程长度（m）
        isw_x0 = 800.0                           # 初始扰动中心位置（m）
        y_center = vs.yt.mean() if hasattr(vs, "yt") else 0.0

        # 存储元数据
        vs.pyc_depth = update(vs.pyc_depth, at[...], pyc_depth)
        vs.isw_amp = update(vs.isw_amp, at[...], isw_amp)

        # 构造网格（Veros 已提供 xt, yt, zt）
        x_1d = vs.xt[:]  # x 中心点
        y_1d = vs.yt[:]  # y 中心点
        z_1d = vs.zt[:]  # z 中心点（通常向下为正，按 veros 约定）

        # 构建背景温度剖面（两层密度结构）
        # 使用 tanh 生成平滑密跃层
        # 温暖表层，冷底层（线性状态方程中温度对应密度）
        t_surf = 15.0
        t_bottom = 5.0

        # 1D 垂向温度剖面
        zcol = z_1d
        # 计算平滑密跃层（tanh）
        temp_profile = t_bottom + 0.5 * (t_surf - t_bottom) * (1.0 - npx.tanh((zcol - pyc_depth) / pyc_thickness))

        # 扩展到三维 (x,y,z)
        temp_3d = npx.broadcast_to(temp_profile[None, None, :], (s.nx, s.ny, s.nz))

        # 加入内孤立波高斯型垂向位移扰动：将位移转成温度扰动（dT/dz * -η）
        # displacement η(x) = A * exp(-(x-x0)^2 / (2 * L^2))
        # dT/dz 近似使用 gradient
        dtemp_dz = npx.gradient(temp_profile, zcol)

        # x 方向位置
        xgrid = vs.xt[:]

        # 高斯型扰动
        xdiff = xgrid - isw_x0
        gauss_x = npx.exp(-0.5 * (xdiff / isw_length) ** 2)

        # 扩展高斯扰动到三维
        gauss_3d = npx.broadcast_to(gauss_x[:, None, None], (s.nx, s.ny, s.nz))

        # 扩展 dT/dz 至 3D
        dTdz_3d = npx.broadcast_to(dtemp_dz[None, None, :], (s.nx, s.ny, s.nz))

        # 温度扰动 = -η * dT/dz
        temp_pert = - (isw_amp * gauss_3d) * dTdz_3d

        # 初始温度场 = 背景 + 扰动
        vs.temp = update(vs.temp, at[...], temp_3d + temp_pert)

        # 盐度：若线性状态方程，可保持常数
        vs.salt = update(vs.salt, at[...], 35.0 * npx.ones_like(vs.temp))

        # 初始速度设为 0
        vs.u = update(vs.u, at[...], npx.zeros_like(vs.u))
        vs.v = update(vs.v, at[...], npx.zeros_like(vs.v))
        vs.w = update(vs.w, at[...], npx.zeros_like(vs.w))

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        # 内孤立波模拟无需外力：完全由初始条件演化
        vs.forc_temp_surface = update(vs.forc_temp_surface, at[...], 0.0)

    @veros_routine
    def set_diagnostics(self, state):
        s = state.settings
        diagnostics = state.diagnostics

        diagnostics["averages"].output_variables = (
            "temp",
            "u",
            "v",
            "w",
            "psi",
        )
        # 内孤立波模拟时间短，设置较高输出频率
        diagnostics["averages"].output_frequency = 3600.0  # 每小时输出一次（可调）
        diagnostics["averages"].sampling_frequency = s.dt_tracer

    @veros_routine
    def after_timestep(self, state):
        # 默认不做任何操作；如需边界缓冲或阻尼可在此添加
        pass

# 如果你想直接使用 `python isw_basic.py` 运行（不推荐），
# Veros 设置不应直接执行，请使用 `veros run`。
