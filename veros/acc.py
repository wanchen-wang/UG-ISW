#!/usr/bin/env python

"""
本 Veros 设置文件由以下命令生成：

   $ veros copy-setup acc_basic

生成时间：2025-10-31 07:07:41 UTC。
"""

__VEROS_VERSION__ = '1.6.0'

if __name__ == "__main__":
    raise RuntimeError(
        "Veros 设置文件不能直接执行。 "
        f"请使用 `veros run {__file__}` 运行。"
    )

# -- 自动生成的头部结束，原始文件内容如下 --


from veros import VerosSetup, veros_routine
from veros.variables import allocate, Variable
from veros.distributed import global_min, global_max
from veros.core.operators import numpy as npx, update, at


class ACCBasicSetup(VerosSetup):
    """一个使用球坐标系的模型，包含部分封闭的海域，用来代表大西洋和南大洋环流（ACC）。

    在通道部分的风应力强迫和表层浮力松弛驱动大尺度的经向翻转环流（MOC）。

    本设置展示了：
     - 如何构建一个理想化几何
     - 如何更新表层强迫
     - 常量水平/垂向混合（关闭 IDEMIX 与 GM_EKE）
     - 诊断输出的基本用法

    :doc:`改编自 ACC 通道模型 </reference/setups/acc>`。
    """

    @veros_routine
    def set_parameter(self, state):
        settings = state.settings
        settings.identifier = "acc_basic"
        settings.description = "我的 ACC 基础设置"

        settings.nx, settings.ny, settings.nz = 30, 42, 15
        settings.dt_mom = 4800                      # 动量方程时间步长
        settings.dt_tracer = 86400 / 2.0           # 示踪物时间步长
        settings.runlen = 86400 * 365 * 20         # 运行 20 年

        settings.x_origin = 0.0
        settings.y_origin = -40.0

        settings.coord_degree = True               # 使用经纬度坐标
        settings.enable_cyclic_x = True            # x 周期边界

        settings.enable_neutral_diffusion = True   # 开启中性扩散
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 500.0
        settings.iso_dslope = 0.005
        settings.iso_slopec = 0.01
        settings.enable_skew_diffusion = True      # 开启倾斜扩散

        settings.enable_hor_friction = True
        settings.A_h = (2 * settings.degtom) ** 3 * 2e-11
        settings.enable_hor_friction_cos_scaling = True
        settings.hor_friction_cosPower = 1

        settings.enable_bottom_friction = True     # 底摩擦
        settings.r_bot = 1e-5

        settings.enable_implicit_vert_friction = True   # 隐式垂向摩擦

        settings.enable_tke = True                 # 开启湍能（TKE）参数化
        settings.c_k = 0.1
        settings.c_eps = 0.7
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 2
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 2e-5
        settings.enable_Prandtl_tke = False
        settings.enable_kappaH_profile = True

        settings.K_gm_0 = 1000.0                   # GM 参数
        settings.enable_eke = False                # 关闭 EKE
        settings.enable_idemix = False             # 关闭内部波混合 IDEMIX

        settings.eq_of_state_type = 3              # 状态方程类型（非线性）

        var_meta = state.var_meta
        var_meta.update(
            t_star=Variable("t_star", ("yt",), "deg C", "参考表层温度"),
            t_rest=Variable("t_rest", ("xt", "yt"), "1/s", "表面温度恢复时间尺度"),
        )

    @veros_routine
    def set_grid(self, state):
        vs = state.variables

        ddz = npx.array(
            [50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0, 390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0]
        )
        vs.dxt = update(vs.dxt, at[...], 2.0)      # x 方向分辨率 2°
        vs.dyt = update(vs.dyt, at[...], 2.0)      # y 方向分辨率 2°
        vs.dzt = ddz[::-1] / 2.5                   # 垂向网格厚度（翻转并缩放）

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        settings = state.settings
        vs.coriolis_t = update(
            vs.coriolis_t, at[:, :],
            2 * settings.omega * npx.sin(vs.yt[None, :] / 180.0 * settings.pi)   # 科氏参数 f = 2Ωsin(y)
        )

    @veros_routine
    def set_topography(self, state):
        vs = state.variables
        x, y = npx.meshgrid(vs.xt, vs.yt, indexing="ij")
        vs.kbot = npx.logical_or(x > 1.0, y < -20).astype("int")   # 简化地形：部分封闭

    @veros_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        # 初始温度：线性递减
        vs.temp = update(vs.temp, at[...], ((1 - vs.zt[None, None, :] / vs.zw[0]) * 15 * vs.maskT)[..., None])
        # 初始盐度：常数 35 PSU
        vs.salt = update(vs.salt, at[...], 35.0 * vs.maskT[..., None])

        # 风应力强迫
        yt_min = global_min(vs.yt.min())
        yu_min = global_min(vs.yu.min())
        yt_max = global_max(vs.yt.max())
        yu_max = global_max(vs.yu.max())

        taux = allocate(state.dimensions, ("yt",))
        # 南大洋风带
        taux = npx.where(vs.yt < -20, 0.1 * npx.sin(settings.pi * (vs.yu - yu_min) / (-20.0 - yt_min)), taux)
        # 北半球风带
        taux = npx.where(vs.yt > 10, 0.1 * (1 - npx.cos(2 * settings.pi * (vs.yu - 10.0) / (yu_max - 10.0))), taux)
        vs.surface_taux = taux * vs.maskU[:, :, -1]

        # 表层热通量强迫
        vs.t_star = allocate(state.dimensions, ("yt",), fill=15)
        vs.t_star = npx.where(vs.yt < -20, 15 * (vs.yt - yt_min) / (-20 - yt_min), vs.t_star)
        vs.t_star = npx.where(vs.yt > 20, 15 * (1 - (vs.yt - 20) / (yt_max - 20)), vs.t_star)
        vs.t_rest = vs.dzt[npx.newaxis, -1] / (30.0 * 86400.0) * vs.maskT[:, :, -1]

        # 湍能表层通量强迫（仅在开启 TKE 时）
        if settings.enable_tke:
            vs.forc_tke_surface = update(
                vs.forc_tke_surface,
                at[2:-2, 2:-2],
                npx.sqrt(
                    (0.5 * (vs.surface_taux[2:-2, 2:-2] + vs.surface_taux[1:-3, 2:-2]) / settings.rho_0) ** 2
                    + (0.5 * (vs.surface_tauy[2:-2, 2:-2] + vs.surface_tauy[2:-2, 1:-3]) / settings.rho_0) ** 2
                )
                ** (1.5),
            )

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        # 表层温度恢复项：Newton cooling
        vs.forc_temp_surface = vs.t_rest * (vs.t_star - vs.temp[:, :, -1, vs.tau])

    @veros_routine
    def set_diagnostics(self, state):
        settings = state.settings
        diagnostics = state.diagnostics

        diagnostics["averages"].output_variables = (
            "salt",
            "temp",
            "u",
            "v",
            "w",
            "psi",
            "surface_taux",
            "surface_tauy",
        )
        diagnostics["averages"].output_frequency = 365 * 86400.0   # 每年输出一次
        diagnostics["averages"].sampling_frequency = settings.dt_tracer * 10

        diagnostics["overturning"].output_frequency = 365 * 86400.0 / 48.0
        diagnostics["overturning"].sampling_frequency = settings.dt_tracer * 10

        diagnostics["tracer_monitor"].output_frequency = 365 * 86400.0 / 12.0

    @veros_routine
    def after_timestep(self, state):
        pass   # 时间步后不进行额外操作
