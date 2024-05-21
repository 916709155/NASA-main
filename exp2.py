import subprocess
import winsound

num_runs = 10  # 指定要执行的次数

for i in range(num_runs):
    # 执行Python程序
    result = subprocess.run(["python", "main2.py"], capture_output=True, text=True)
    print(f"第{i + 1}次执行完成！")
    print(f"Standard output:{result.stdout}")
winsound.Beep(1000, 500)  # 播放一次响铃声，持续时间为500毫秒
