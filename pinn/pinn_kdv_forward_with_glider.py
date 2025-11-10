# pinn_kdv_forward_with_glider.py
# 说明：PyTorch PINN，用合成解析解演示“正问题 + 滑翔机垂直速度监督”
# 运行环境建议：Python 3.8+, PyTorch，最好有 GPU。将此脚本保存后直接运行。

import torch, numpy as np, math, matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def sech_torch(x): return 1.0 / torch.cosh(x)

def analytic_eta_and_params(x, t, params):
    # x,t: torch tensors (N,1) or numpy arrays convertible to tensor
    h2 = float(params['h2']); h1 = float(params['h1'])
    rho2 = float(params['rho2']); rho1 = float(params['rho1'])
    a = float(params['a']); g = float(params['g'])
    inner = g * h1 * h2 * (rho1 - rho2) / (rho1 * h2 + rho2 * h1 + 1e-12)
    c0 = math.sqrt(abs(inner) + 1e-12)   # 修正：abs 避免负根号
    c1 = -3.0 * c0 * (rho2 * h2**2 - rho1 * h2**2) / (rho2 * h2 * h1**2 + rho1 * h1 * h2**2 + 1e-12)
    c2 = c0 * (1.0 / 6.0) * (rho2 * h2**2 * h1 + rho1 * h2**2 * h2) / (rho2 * h1 + rho1 * h2 + 1e-12)
    lam = math.sqrt(max(1e-8, 12.0 * c2 / (a * c1 + 1e-12)))
    c = c0 + a * c1 / 3.0
    # compute eta
    Z = (x - c * t) / lam
    eta = a * sech_torch(Z)**2
    return eta, c0, c1, c2, c, lam

# ----- PINN 网络 -----
class PINN(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i+1]))
        self.act = torch.tanh
        for m in self.layers:
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None: m.bias.data.fill_(0.0)
    def forward(self, xt):
        # xt: (N,2) with columns [x,t]
        y = xt
        for layer in self.layers[:-1]:
            y = layer(y); y = self.act(y)
        y = self.layers[-1](y)
        return y

# ----- PDE 残差（注意：x, t 以独立叶子张量输入，便于对 t 求导） -----
def pde_residual(model, x_in, t_in, c0, c1, c2):
    xt = torch.cat([x_in, t_in], dim=1)
    eta = model(xt)
    eta_t = torch.autograd.grad(eta, t_in, grad_outputs=torch.ones_like(eta), create_graph=True, retain_graph=True)[0]
    eta_x = torch.autograd.grad(eta, x_in, grad_outputs=torch.ones_like(eta), create_graph=True, retain_graph=True)[0]
    eta_xx = torch.autograd.grad(eta_x, x_in, grad_outputs=torch.ones_like(eta_x), create_graph=True, retain_graph=True)[0]
    eta_xxx = torch.autograd.grad(eta_xx, x_in, grad_outputs=torch.ones_like(eta_xx), create_graph=True, retain_graph=True)[0]
    res = eta_t + c0*eta_x + c1*eta*eta_x + c2*eta_xxx
    return res

# ---------------- 生成合成数据（演示） ----------------
params = {'h2':1.0, 'h1':4.0, 'rho2':1.000, 'rho1':0.998, 'a':-0.08, 'g':9.81}
x_min,x_max = -40.0,40.0; t_min,t_max = 0.0,4.0

# collocation 点
N_r = 4000
Xr = np.hstack([np.random.uniform(x_min,x_max,(N_r,1)), np.random.uniform(t_min,t_max,(N_r,1))])
X_r_t = torch.tensor(Xr, dtype=torch.float32, device=device)

# 边界（远场） x = +/- domain
N_bc = 200
t_bc = np.random.uniform(t_min,t_max,(N_bc,1))
X_bc = np.vstack([np.hstack([np.full((N_bc,1),x_min), t_bc]), np.hstack([np.full((N_bc,1),x_max), t_bc])])
X_bc_t = torch.tensor(X_bc, dtype=torch.float32, device=device)
eta_bc = torch.zeros((X_bc_t.shape[0],1), dtype=torch.float32, device=device)

# 稀疏场数据（eta）
N_data = 1000
Xd = np.hstack([np.random.uniform(-20,20,(N_data,1)), np.random.uniform(t_min,t_max,(N_data,1))])
X_d_t = torch.tensor(Xd, dtype=torch.float32, device=device)
eta_exact, c0_val, c1_val, c2_val, c_val, lam_val = analytic_eta_and_params(X_d_t[:,0:1], X_d_t[:,1:2], params)
eta_exact = eta_exact.to(device)

# ----- 滑翔机轨迹与垂直速度监督 -----
N_glider = 200
t_glider = np.linspace(t_min,t_max,N_glider)[:,None]
x0 = -10.0; U_glider = 2.0
x_glider = x0 + U_glider * t_glider
X_glider = np.hstack([x_glider, t_glider]); X_glider_t = torch.tensor(X_glider, dtype=torch.float32, device=device)

# 用解析公式计算 eta_t（避免 autograd 图依赖问题）：
# eta = a * sech(Z)^2, Z=(x - c t)/lam
# deta/dt = 2 * a * c / lam * sech(Z)^2 * tanh(Z)
_,_,_,_, c_scalar, lam_scalar = analytic_eta_and_params(X_glider_t[:,0:1], X_glider_t[:,1:2], params)
xg = X_glider_t[:,0:1]; tg = X_glider_t[:,1:2]
Z = (xg - c_scalar * tg) / lam_scalar
sechZ = 1.0 / torch.cosh(Z); tanhZ = torch.tanh(Z)
w_g = 2.0 * params['a'] * c_scalar / lam_scalar * (sechZ**2) * tanhZ
w_obs = w_g.to(device)   # 这是滑翔机处的监督量（合成），真实数据请替换

# ----- 模型 + 优化器 -----
layers = [2,40,40,40,40,1]
model = PINN(layers).to(device)
c0 = torch.tensor(c0_val, dtype=torch.float32, device=device)
c1 = torch.tensor(c1_val, dtype=torch.float32, device=device)
c2 = torch.tensor(c2_val, dtype=torch.float32, device=device)

wpde=0.1; wbc=1.0; wdata=10.0; wglider=10.0
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)

# ----- 训练循环（注意：对 glider 和 PDE 残差，我们把 x,t 做为 leaf requires_grad True） -----
def train(n_iter=5000, print_every=500):
    for it in range(1, n_iter+1):
        model.train(); optimizer.zero_grad()
        pred_d = model(X_d_t); loss_data = torch.mean((pred_d - eta_exact)**2)
        # glider supervision: make xg_var,tg_var leaf tensors
        xg_var = X_glider_t[:,0:1].clone().detach().requires_grad_(True)
        tg_var = X_glider_t[:,1:2].clone().detach().requires_grad_(True)
        pred_g = model(torch.cat([xg_var, tg_var], dim=1))
        eta_t_g = torch.autograd.grad(pred_g, tg_var, grad_outputs=torch.ones_like(pred_g), create_graph=True)[0]
        loss_glider = torch.mean((eta_t_g - w_obs)**2)
        # PDE residual (make collocation leaf x/t)
        xr_var = X_r_t[:,0:1].clone().detach().requires_grad_(True)
        tr_var = X_r_t[:,1:2].clone().detach().requires_grad_(True)
        res = pde_residual(model, xr_var, tr_var, c0, c1, c2)
        loss_pde = torch.mean(res**2)
        # boundary
        pred_bc = model(X_bc_t); loss_bc = torch.mean((pred_bc - eta_bc)**2)
        loss = wpde*loss_pde + wbc*loss_bc + wdata*loss_data + wglider*loss_glider
        loss.backward(); optimizer.step(); scheduler.step()
        if it % print_every == 0 or it == 1:
            print(f"Iter {it:5d} | Loss {loss.item():.3e} | pde {loss_pde.item():.3e} | data {loss_data.item():.3e} | glider {loss_glider.item():.3e} | bc {loss_bc.item():.3e}")

# 运行训练（本地若有 GPU，建议 n_iter=20000+）
if __name__ == "__main__":
    train(n_iter=5000, print_every=500)
    # 评估：x=0 切片
    # ===================== 绘图部分（修正版） =====================

    # 构造测试点：x固定为0，t从0~4秒
    t_test = np.linspace(t_min, t_max, 301)[:, None]
    x_test = np.zeros_like(t_test)
    X_test_t = torch.tensor(np.hstack([x_test, t_test]), dtype=torch.float32, device=device)

    # PINN预测
    model.eval()
    with torch.no_grad():
        eta_pred = model(X_test_t).cpu().numpy().flatten()

    # 正确计算解析解（使用torch运算，保持时间依赖）
    eta_true, _, _, _, _, _ = analytic_eta_and_params(
        X_test_t[:, 0:1],  # x = 0
        X_test_t[:, 1:2],  # t in [0,4]
        params
    )
    eta_true = eta_true.cpu().numpy().flatten()

    # 绘制结果
    plt.figure(figsize=(9,4))
    plt.plot(t_test.flatten(), eta_true, label='Analytic (true)', linewidth=2)
    plt.plot(t_test.flatten(), eta_pred, '--', label='PINN prediction', linewidth=2)
    plt.title('Interface displacement η(t) at x=0')
    plt.xlabel('Time t (s)')
    plt.ylabel('Interface displacement η (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制误差
    plt.figure(figsize=(9,3))
    plt.plot(t_test.flatten(), np.abs(eta_true - eta_pred))
    plt.title('Absolute error |η_true - η_pred|')
    plt.xlabel('Time t (s)')
    plt.ylabel('Error (m)')
    plt.grid(True)
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "pinn_kdv_forward_with_glider.pth")
    print("Saved model to pinn_kdv_forward_with_glider.pth")
