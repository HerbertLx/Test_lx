import numpy as np
import matplotlib.pyplot as plt
import os


def get_sigma(t, sigma_init=1.0, sigma_final=0.1, T=500000):
    """
    根据步数 t 线性衰减噪声标准差 sigma。
    """
    # 计算衰减比例，确保不小于 0 且不大于 1
    decay_ratio = min(t / T, 1.0)
    # 线性插值计算当前 sigma
    sigma_t = sigma_init + (1.0 - decay_ratio) * (sigma_final - sigma_init)
    return sigma_t


def plot_sigma_curve(
    sigma_init=1.0,
    sigma_final=0.1,
    T=500000,
    num_points=1000,
    save_dir="/home/cuhk/Documents/Test_lx/DrQ_v2/result",
    filename="sigma_decay_curve.png",
):
    """
    绘制 get_sigma 随步数 t 变化的曲线，并将图像保存到指定目录。
    """
    # 生成步数
    t_values = np.linspace(0, 2*T, num_points)
    # 计算对应的 sigma
    sigma_values = [get_sigma(t, sigma_init, sigma_final, T) for t in t_values]

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(t_values, sigma_values, label="sigma(t)")
    plt.xlabel("t (steps)")
    plt.ylabel("sigma")
    plt.title("Sigma Decay Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path)
    plt.close()

    print(f"图像已保存到: {save_path}")


if __name__ == "__main__":
    # 运行脚本时自动生成并保存图像
    plot_sigma_curve()
