import matplotlib.pyplot as plt
import nidaqmx
import csv
from datetime import datetime

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.ion()

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
lines = []
for ax in axs.flat:
    line, = ax.plot([], [])
    lines.append(line)
    ax.set_xlim(0, 2000)
    ax.set_ylim(-0.1, 0.1)
    ax.grid(True)

# 타이틀
axs[0, 0].set_title("sensor1 (ai0)")
axs[0, 1].set_title("sensor2 (ai1)")
axs[1, 0].set_title("sensor3 (ai2)")
axs[1, 1].set_title("sensor4 (ai3)")

# ⏳ 현재 시간 기준 파일명
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"pvdf_data_back_carbon {timestamp}.csv"

# CSV 파일 생성
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sensor1", "sensor2", "sensor3", "sensor4"])  # 헤더

    try:
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan("cDAQ1Mod3/ai0:3")
            task.timing.cfg_samp_clk_timing(
                rate=100000,
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=2000
            )

            while True:
                data = task.read(number_of_samples_per_channel=2000)  # shape: [4][2000]

                # 그래프 업데이트
                for i in range(4):
                    lines[i].set_ydata(data[i])
                    lines[i].set_xdata(range(len(data[i])))
                    axs.flat[i].set_ylim(-0.1, 0.1)

                fig.tight_layout()
                fig.canvas.draw()
                fig.canvas.flush_events()

                # CSV에 샘플 저장 (row-wise로 transpose)
                for row in zip(*data):  # 2000행짜리 4열로 저장됨
                    writer.writerow(row)

                print("샘플 저장 완료 - 각 센서 peak:",
                      [max(abs(x) for x in ch) for ch in data])

    except KeyboardInterrupt:
        print("\n종료: 사용자에 의해 중단되었습니다.")
        plt.close('all')
