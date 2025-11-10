import xarray as xr
import matplotlib.pyplot as plt

# 打开仿真结果
ds = xr.open_dataset("acc_basic.averages.nc")

print(ds)  # 查看有哪些变量

# 举例：绘制某个时间步的温度场
temp = ds['temp'].isel(time=-1, z=-1)  # 取最后一个时间步、最浅一层
plt.figure(figsize=(8,4))
temp.plot(cmap='coolwarm')
plt.title("Surface Temperature (last timestep)")
plt.show()

# 举例：绘制垂向剖面（经度方向取中间一列）
temp_section = ds['temp'].isel(time=-1, x=slice(None, None, 2))[:, ds.y.size//2]
plt.figure(figsize=(6,4))
temp_section.plot(y='z', cmap='RdYlBu_r')
plt.title("Temperature vertical section at mid-latitude")
plt.gca().invert_yaxis()
plt.show()

# 举例：绘制流速（矢量图）
u = ds['u'].isel(time=-1, z=-1)
v = ds['v'].isel(time=-1, z=-1)
plt.figure(figsize=(8,4))
plt.quiver(u['x'], u['y'], u, v, scale=10)
plt.title("Surface velocity field")
plt.show()
