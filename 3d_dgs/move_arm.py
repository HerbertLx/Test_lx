import sys
import os
import math
import time
import numpy as np

# 1. 动态设置路径以引用 KinovaManager
current_dir = os.path.dirname(os.path.abspath(__file__))
# 找到 /home/cuhk/Documents/ 这一层
parent_dir = os.path.dirname(os.path.dirname(current_dir)) 
lib_path = os.path.join(parent_dir, 'KinovaGen3_Easy_Control')

if lib_path not in sys.path:
    sys.path.append(lib_path)

from kinova_manage import KinovaManager

def run_velocity_circle():
    # 初始化并连接
    arm = KinovaManager(ip_address="192.168.8.10")
    arm.connect()

    try:
        # 2. 初始定位 (使用笛卡尔绝对位置)
        print("正在移动到初始位置...")
        # [X, Y, Z, ThetaX, ThetaY, ThetaZ, Gripper]
        initial_pose = [0.25, 0.1, 0.3, 135.0, -90.0, 132.0, 100.0]
        arm.move_cartesian(initial_pose, dual_grip=False)
        time.sleep(2.0) 

        # 3. 圆周运动参数设置
        radius = 0.1
        linear_speed = 0.05  # 设定线速度为 0.05 m/s
        omega = linear_speed / radius  # 计算角速度 (rad/s)
        
        # 运行参数
        total_time = (2 * math.pi) / omega  # 完成一圈所需时间
        dt = 0.01  # 控制频率 (100Hz)
        start_time = time.time()
        
        print(f"开始速度控制圆周运动，预计耗时: {total_time:.2f}秒")

        while True:
            elapsed_time = time.time() - start_time
            # if elapsed_time > total_time:
            #     break
            
            # 当前角度 (从 pi 开始，顺时针或逆时针取决于 omega 正负)
            # 这里设为逆时针旋转
            theta = math.pi + (omega * elapsed_time)
            
            # 计算速度分量
            # Vx = -R * sin(theta) * omega
            # Vy =  R * cos(theta) * omega
            vx = -radius * math.sin(theta) * omega
            vy = radius * math.cos(theta) * omega
            
            # 发送速度指令 [Vx, Vy, Vz, Wx, Wy, Wz]
            # 保持姿态角速度为 0，Z轴速度为 0
            speeds = [vx, vy, 0.0, 0.0, 0.0, 0.0]
            
            # duration_ms 设为较小值（如100ms），作为心跳安全机制
            arm.move_velocity(speeds)
            
            time.sleep(dt)

        # 4. 停止运动
        print("运动完成，停止机械臂。")
        arm.move_velocity([0, 0, 0, 0, 0, 0])

    except KeyboardInterrupt:
        print("\n用户中断，停止机械臂。")
        arm.move_velocity([0, 0, 0, 0, 0, 0])
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        arm.disconnect()

if __name__ == "__main__":
    run_velocity_circle()