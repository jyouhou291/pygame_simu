# =============================================================================
# 歩きスマホ行動シミュレーション (v10.0 - Analysis Plot)
# =============================================================================
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os
from collections import deque

class AnalysisPlot:
    def __init__(self, root):
        self.root = root
        self.root.title("リアルタイム分析")
        self.root.geometry("600x800") # 高さを広げる

        # データ保持用のdeque
        self.max_len = 100
        self.time_data = deque(maxlen=self.max_len)
        self.speed_data = deque(maxlen=self.max_len)
        self.agent_count_data = deque(maxlen=self.max_len)

        # --- GUI要素 --- 
        # 上部にテキスト情報を表示するフレーム
        info_frame = tk.Frame(root, bg='#f0f0f0')
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        self.total_collisions_var = tk.StringVar(value="総衝突回数: 0")
        total_collisions_label = tk.Label(info_frame, textvariable=self.total_collisions_var, font=('Meiryo UI', 12, 'bold'), bg='#f0f0f0')
        total_collisions_label.pack()

        # MatplotlibのFigureとAxesを準備
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(6, 7.5), tight_layout=True, gridspec_kw={'height_ratios': [1, 1, 1.2]})
        self.fig.patch.set_facecolor('#f0f0f0')

        # プロット1 (平均速度)
        self.line1, = self.ax1.plot([], [], 'b-', label="平均速度 (m/s)")
        self.ax1.set_ylabel("速度 (m/s)")
        self.ax1.legend(loc='upper left')
        self.ax1.grid(True)

        # プロット2 (人数)
        self.line2, = self.ax2.plot([], [], 'g-', label="現在人数")
        self.ax2.set_ylabel("人数")
        self.ax2.legend(loc='upper left')
        self.ax2.grid(True)

        # プロット3 (タイプ別衝突率 - 棒グラフ)
        self.bar_labels = ['通常歩行', '歩きスマホ']
        self.bar_colors = ['blue', 'red']
        self.bars = self.ax3.bar(self.bar_labels, [0, 0], color=self.bar_colors)
        self.ax3.set_ylabel('一人当たりの衝突回数')
        self.ax3.set_title('危険性の比較')

        # TkinterにMatplotlibのCanvasを埋め込む
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # アニメーションを開始
        self.ani = FuncAnimation(self.fig, self.animate, interval=1000, blit=False)

    def animate(self, i):
        try:
            filepath = os.path.join(os.path.dirname(__file__), "live_data.json")
            if not os.path.exists(filepath): return

            with open(filepath, "r") as f:
                data = json.load(f)

            # テキスト情報を更新
            self.total_collisions_var.set(f"総衝突回数: {data.get('total_collisions', 0)}")

            # 時系列データを更新
            self.time_data.append(data.get("time", 0))
            self.speed_data.append(data.get("avg_speed", 0))
            self.agent_count_data.append(data.get("agent_count", 0))

            # 時系列プロットを更新
            self.line1.set_data(self.time_data, self.speed_data)
            self.line2.set_data(self.time_data, self.agent_count_data)
            self.ax1.relim(); self.ax1.autoscale_view()
            self.ax2.relim(); self.ax2.autoscale_view()

            # 棒グラフデータを更新
            rate_normal = data.get("collision_rate_normal", 0)
            rate_phone = data.get("collision_rate_phone", 0)
            self.bars[0].set_height(rate_normal)
            self.bars[1].set_height(rate_phone)
            self.ax3.relim(); self.ax3.autoscale_view()
            max_rate = max(0.1, rate_normal, rate_phone)
            self.ax3.set_ylim(0, max_rate * 1.2)

        except (json.JSONDecodeError, FileNotFoundError): pass
        except Exception as e: print(f"プロット更新中にエラーが発生しました: {e}")

if __name__ == '__main__':
    plt.rcParams['font.family'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'sans-serif']
    plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'MS Gothic']
    root = tk.Tk()
    app = AnalysisPlot(root)
    root.mainloop()