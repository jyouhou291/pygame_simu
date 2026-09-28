# =============================================================================
# 歩きスマホ行動シミュレーション (v12.0 - Control Panel)
# =============================================================================
import tkinter as tk
from tkinter import ttk
import subprocess
import os

class ControlPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("シミュレーション・コントロールパネル")
        self.simulation_process = None
        self.analysis_process = None

        style = ttk.Style()
        style.configure('TLabel', font=('Meiryo UI', 10))
        style.configure('TButton', font=('Meiryo UI', 10))
        style.configure('TRadiobutton', font=('Meiryo UI', 10))
        style.configure('TFrame', background='#f0f0f0')

        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- パラメータ設定 ---
        params_frame = ttk.LabelFrame(main_frame, text="パラメータ設定", padding="10")
        params_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)

        self.p_phone_var = tk.DoubleVar(value=30.0)
        ttk.Label(params_frame, text="歩きスマホ率 (%)").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.p_phone_scale = ttk.Scale(params_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.p_phone_var, length=250)
        self.p_phone_scale.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E))
        self.p_phone_label = ttk.Label(params_frame, text="30.0 %")
        self.p_phone_label.grid(row=0, column=1, columnspan=2, sticky=tk.E)

        self.spawn_rate_var = tk.DoubleVar(value=0.5)
        ttk.Label(params_frame, text="生成レート (人/秒)").grid(row=2, column=0, sticky=tk.W, pady=(10, 2))
        self.spawn_rate_scale = ttk.Scale(params_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, variable=self.spawn_rate_var, length=250)
        self.spawn_rate_scale.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E))
        self.spawn_rate_label = ttk.Label(params_frame, text="1.0 人/秒")
        self.spawn_rate_label.grid(row=2, column=1, columnspan=2, sticky=tk.E)

        # --- 再生速度 --- 
        speed_frame = ttk.LabelFrame(main_frame, text="再生速度", padding="10")
        speed_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.speed_var = tk.IntVar(value=1)
        ttk.Radiobutton(speed_frame, text="1x", variable=self.speed_var, value=1).pack(side=tk.LEFT, padx=20)
        ttk.Radiobutton(speed_frame, text="2x", variable=self.speed_var, value=2).pack(side=tk.LEFT, padx=20)
        ttk.Radiobutton(speed_frame, text="4x", variable=self.speed_var, value=4).pack(side=tk.LEFT, padx=20)

        # --- 操作ボタン ---
        actions_frame = ttk.LabelFrame(main_frame, text="操作", padding="10")
        actions_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.research_button = ttk.Button(actions_frame, text="論文設定を適用", command=self.apply_research_settings)
        self.research_button.pack(fill=tk.X, padx=5, pady=3)
        self.start_button = ttk.Button(actions_frame, text="シミュレーション開始 / 再起動", command=self.start_or_restart_simulation)
        self.start_button.pack(fill=tk.X, padx=5, pady=3)
        self.stop_button = ttk.Button(actions_frame, text="シミュレーション停止", command=self.stop_all_processes)
        self.stop_button.pack(fill=tk.X, padx=5, pady=3)

        self.status_var = tk.StringVar(value="状態: 待機中")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=(10, 0))

        self.p_phone_var.trace_add("write", self.update_labels)
        self.spawn_rate_var.trace_add("write", self.update_labels)
        self.update_labels()

    def update_labels(self, *args):
        self.p_phone_label.config(text=f"{self.p_phone_var.get():.1f} %")
        self.spawn_rate_label.config(text=f"{self.spawn_rate_var.get():.1f} 人/秒")

    def apply_research_settings(self):
        self.p_phone_var.set(15.0); self.spawn_rate_var.set(0.8)
        self.status_var.set("状態: 論文設定を適用。開始ボタンを押してください。")

    def start_or_restart_simulation(self):
        self.stop_all_processes()
        self.status_var.set("状態: シミュレーションを開始中...")
        p_phone = self.p_phone_var.get() / 100.0
        spawn_rate = self.spawn_rate_var.get()
        speed = self.speed_var.get()

        sim_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation.py")
        analysis_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_plot.py")
        
        sim_command = ["python", sim_script_path, f"--p_phone={p_phone}", f"--spawn_rate={spawn_rate}", f"--speed={speed}"]
        analysis_command = ["python", analysis_script_path]

        try:
            self.simulation_process = subprocess.Popen(sim_command)
            self.analysis_process = subprocess.Popen(analysis_command)
            self.status_var.set(f"状態: シミュレーション実行中 (x{speed})")
        except FileNotFoundError as e:
            self.status_var.set(f"エラー: {e.filename} が見つかりません。")
        except Exception as e:
            self.status_var.set(f"エラー: {e}")

    def stop_all_processes(self):
        stopped = False
        if self.simulation_process and self.simulation_process.poll() is None:
            self.simulation_process.terminate(); self.simulation_process.wait(); stopped = True
        if self.analysis_process and self.analysis_process.poll() is None:
            self.analysis_process.terminate(); self.analysis_process.wait(); stopped = True
        if stopped:
            self.status_var.set("状態: プロセスを停止しました。")

    def on_closing(self):
        self.stop_all_processes()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = ControlPanel(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
