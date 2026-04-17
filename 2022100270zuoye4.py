"""
- FFT方法：找到包含95%能量的最高频率
- 梯度方法：用空域梯度估计局部最高频率
- 对比两者一致性
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import sys
import os

# ── 中文字体支持 ──────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ═══════════════════════════════════════════════════════════════
# 1. FFT 方法：找到包含 95% 能量的最高频率
# ═══════════════════════════════════════════════════════════════
def fft_max_freq(block, energy_ratio=0.95):
    """
    对单个图像块做2D FFT，返回包含 energy_ratio 能量的最高频率（归一化，0~0.5）。

    流程：I(x,y) --FFT--> F[k] --功率谱--> P[k] --积分判断--> f_rms
    这里我们用"累积能量截断"而非 f_rms，更直观地对应"最高频率"。
    """
    H, W = block.shape
    # 2D FFT + 移频（DC 居中）
    F = np.fft.fft2(block)
    F_shift = np.fft.fftshift(F)
    P = np.abs(F_shift) ** 2  # 功率谱

    # 构建每个频率分量到中心（DC）的归一化距离
    cy, cx = H // 2, W // 2
    v = np.fft.fftshift(np.fft.fftfreq(H))  # 归一化频率 -0.5~0.5
    u = np.fft.fftshift(np.fft.fftfreq(W))
    UU, VV = np.meshgrid(u, v)
    R = np.sqrt(UU**2 + VV**2)  # 径向频率

    # 按频率从小到大排序，累积能量
    idx = np.argsort(R.ravel())
    r_sorted = R.ravel()[idx]
    p_sorted = P.ravel()[idx]
    cum_energy = np.cumsum(p_sorted)
    total = cum_energy[-1]
    if total == 0:
        return 0.0

    # 找到累积能量刚超过阈值的频率
    threshold_idx = np.searchsorted(cum_energy, energy_ratio * total)
    threshold_idx = min(threshold_idx, len(r_sorted) - 1)
    f_max_fft = r_sorted[threshold_idx]
    return float(f_max_fft)


# ═══════════════════════════════════════════════════════════════
# 2. 梯度方法：用空域梯度估计局部最高频率
# ═══════════════════════════════════════════════════════════════
def gradient_max_freq(block):
    """
    空域梯度方法：
    流程：I(x,y) --空域差分--> |∇I| --统计--> E[|∇I|²] --÷4π²Var(I)--> f_rms²

    f_rms = sqrt( E[|∇I|²] / (4π² · Var(I)) )
    单位：归一化频率（0~0.5）
    """
    block = block.astype(np.float64)

    # 有限差分（中心差分）
    gy = np.diff(block, axis=0, prepend=block[[0], :])
    gx = np.diff(block, axis=1, prepend=block[:, [0]])
    grad_sq = gx**2 + gy**2

    mean_grad_sq = np.mean(grad_sq)
    var_I = np.var(block)

    if var_I < 1e-8:
        return 0.0

    # f_rms（归一化）
    f_rms = np.sqrt(mean_grad_sq / (4 * np.pi**2 * var_I))
    # 归一化频率上限约为 0.5（Nyquist）
    f_rms = min(f_rms, 0.5)
    return float(f_rms)


# ═══════════════════════════════════════════════════════════════
# 3. 主函数：分块 + 对比
# ═══════════════════════════════════════════════════════════════
def analyze_image(image_path, block_size=32, energy_ratio=0.95):
    """
    对整幅图像分块，逐块计算 FFT 频率和梯度频率，对比一致性。
    """
    # 读取图像（灰度）
    img = Image.open(image_path).convert('L')
    img_arr = np.array(img, dtype=np.float64)
    H, W = img_arr.shape
    print(f"图像尺寸: {W} x {H}，块大小: {block_size}x{block_size}")

    rows = H // block_size
    cols = W // block_size

    fft_map   = np.zeros((rows, cols))
    grad_map  = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            block = img_arr[r*block_size:(r+1)*block_size,
                            c*block_size:(c+1)*block_size]
            fft_map[r, c]  = fft_max_freq(block, energy_ratio)
            grad_map[r, c] = gradient_max_freq(block)

    return img_arr, fft_map, grad_map, block_size


def plot_results(img_arr, fft_map, grad_map, block_size, save_path=None):
    """可视化结果"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    vmin = 0
    vmax = max(fft_map.max(), grad_map.max(), 0.01)

    # ── 原图 ──
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_arr, cmap='gray')
    ax0.set_title('原始图像（灰度）', fontsize=12)
    ax0.axis('off')

    # ── FFT 频率图 ──
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(fft_map, cmap='hot', vmin=vmin, vmax=vmax)
    ax1.set_title(f'FFT方法：95%能量最高频率\n(块大小={block_size})', fontsize=11)
    ax1.set_xlabel('列块索引'); ax1.set_ylabel('行块索引')
    plt.colorbar(im1, ax=ax1, label='归一化频率')

    # ── 梯度频率图 ──
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(grad_map, cmap='hot', vmin=vmin, vmax=vmax)
    ax2.set_title(f'梯度方法：f_rms 估计频率\n(块大小={block_size})', fontsize=11)
    ax2.set_xlabel('列块索引'); ax2.set_ylabel('行块索引')
    plt.colorbar(im2, ax=ax2, label='归一化频率')

    # ── 差异图 ──
    diff_map = fft_map - grad_map
    ax3 = fig.add_subplot(gs[1, 0])
    absmax = np.abs(diff_map).max() or 0.01
    im3 = ax3.imshow(diff_map, cmap='RdBu_r', vmin=-absmax, vmax=absmax)
    ax3.set_title('差异图 (FFT − 梯度)', fontsize=11)
    ax3.set_xlabel('列块索引'); ax3.set_ylabel('行块索引')
    plt.colorbar(im3, ax=ax3, label='频率差')

    # ── 散点图：一致性分析 ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(grad_map.ravel(), fft_map.ravel(), alpha=0.4, s=12, c='steelblue')
    lim = max(vmax, 0.01)
    ax4.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='y=x（完全一致）')
    ax4.set_xlabel('梯度方法 f_rms')
    ax4.set_ylabel('FFT方法 f_max(95%)')
    ax4.set_title('两种方法一致性散点图', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.set_xlim(0, lim); ax4.set_ylim(0, lim)

    # 计算相关系数
    corr = np.corrcoef(fft_map.ravel(), grad_map.ravel())[0, 1]
    ax4.text(0.05, 0.92, f'Pearson r = {corr:.3f}',
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # ── 直方图对比 ──
    ax5 = fig.add_subplot(gs[1, 2])
    bins = np.linspace(0, vmax + 0.01, 30)
    ax5.hist(fft_map.ravel(),  bins=bins, alpha=0.6, label='FFT方法',  color='tomato')
    ax5.hist(grad_map.ravel(), bins=bins, alpha=0.6, label='梯度方法', color='steelblue')
    ax5.set_xlabel('归一化频率'); ax5.set_ylabel('块数量')
    ax5.set_title('频率分布直方图对比', fontsize=11)
    ax5.legend()

    fig.suptitle('图像分块频率分析：FFT方法 vs 梯度方法', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存至: {save_path}")
    plt.show()


def print_statistics(fft_map, grad_map):
    """打印统计摘要"""
    diff = fft_map - grad_map
    corr = np.corrcoef(fft_map.ravel(), grad_map.ravel())[0, 1]
    print("\n" + "="*55)
    print("           频率分析统计摘要")
    print("="*55)
    print(f"{'指标':<20} {'FFT方法':>12} {'梯度方法':>12}")
    print("-"*55)
    print(f"{'均值':<20} {fft_map.mean():>12.4f} {grad_map.mean():>12.4f}")
    print(f"{'标准差':<20} {fft_map.std():>12.4f} {grad_map.std():>12.4f}")
    print(f"{'最大值':<20} {fft_map.max():>12.4f} {grad_map.max():>12.4f}")
    print(f"{'最小值':<20} {fft_map.min():>12.4f} {grad_map.min():>12.4f}")
    print("-"*55)
    print(f"两者 Pearson 相关系数: {corr:.4f}")
    print(f"平均绝对误差 (MAE):    {np.abs(diff).mean():.4f}")
    print(f"均方根误差 (RMSE):     {np.sqrt((diff**2).mean()):.4f}")
    print("="*55)
    if corr > 0.8:
        print("✅ 两种方法高度一致（r > 0.8）")
    elif corr > 0.5:
        print("⚠️  两种方法中等一致（0.5 < r ≤ 0.8）")
    else:
        print("❌ 两种方法一致性较低（r ≤ 0.5）")
    print()


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # ── 参数 ──────────────────────────────────────────────────
    # 用法：python freq_analysis.py [图像路径] [块大小] [能量比例]
    image_path   = sys.argv[1] if len(sys.argv) > 1 else 'test.jpg'
    block_size   = int(sys.argv[2])   if len(sys.argv) > 2 else 32
    energy_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95

    if not os.path.exists(image_path):
        # 若无图像，生成一张合成测试图
        print(f"未找到 '{image_path}'，生成合成测试图像...")
        np.random.seed(0)
        H, W = 256, 256
        x = np.linspace(0, 4*np.pi, W)
        y = np.linspace(0, 4*np.pi, H)
        XX, YY = np.meshgrid(x, y)
        synthetic = (
            128 +
            50  * np.sin(2*XX) * np.cos(3*YY) +   # 低频
            30  * np.sin(15*XX) +                   # 中频
            10  * np.random.randn(H, W)             # 噪声（高频）
        ).clip(0, 255).astype(np.uint8)
        Image.fromarray(synthetic).save('test.jpg')
        image_path = 'test.jpg'

    # ── 分析 ──────────────────────────────────────────────────
    img_arr, fft_map, grad_map, bs = analyze_image(
        image_path, block_size=block_size, energy_ratio=energy_ratio
    )

    print_statistics(fft_map, grad_map)

    save_path = os.path.splitext(image_path)[0] + '_freq_analysis.png'
    plot_results(img_arr, fft_map, grad_map, bs, save_path=save_path)
