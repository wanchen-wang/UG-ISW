import xarray as xr
import matplotlib.pyplot as plt

# 1. 打开 Veros 输出文件
ds = xr.open_dataset("acc_basic.averages.nc")
print(ds)

# 2. 绘制表层温度场
temp_surface = ds['temp'].isel(Time=-1, zt=-1)
plt.figure(figsize=(8, 4))
temp_surface.plot(cmap='turbo')
plt.title('Surface Temperature (ACC Basic)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('acc_surface_temp.png', dpi=200)
plt.show()

# 3. 绘制经向速度剖面（取中间一条经线）
v_slice = ds['v'].isel(Time=-1, xt=len(ds.xt)//2)
plt.figure(figsize=(6, 4))
v_slice.plot(y='zt', cmap='RdBu_r', robust=True)
plt.title('Meridional Velocity Section (mid-longitude)')
plt.ylabel('Depth (m)')
plt.tight_layout()
plt.savefig('acc_v_section.png', dpi=200)
plt.show()
