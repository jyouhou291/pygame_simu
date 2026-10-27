# =============================================================================
# 歩きスマホ行動シミュレーション ― 共通物理エンジン (sim_core.py)
#
# 2D版 (simulation.py, Pygame) と 3D版 (simulation_3d.py, Ursina) の両方が
# この同じ物理ロジックを使う。レンダラー（描画方法）に依存する処理は一切
# 含まない（numpyのみに依存）。衝突・視野・反応遅延などの計算ロジックを
# 二重管理しないためにここへ切り出した。
#
# 参考文献は simulation.py の冒頭コメントを参照。
# =============================================================================
from __future__ import annotations
import numpy as np
import json
import csv
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


@dataclass
class Config:
    # --- 歩道空間の定義（二方向のすれ違い） ---
    # SIDEWALK_HEIGHT_M: 「人二人がすれ違えるくらいの日本の道」幅（1.5m）に設定。
    # 元の2.5mは欧米的な広い歩道の感覚に近かったため、より日本の街らしい狭さにした。
    WIDTH_M: float = 30.0; SIDEWALK_HEIGHT_M: float = 1.5; ROAD_HEIGHT_M: float = 4.0
    LAMPPOST_RADIUS_M: float = 0.15  # 街灯・標識ポール（障害物）の太さ

    AGENT_RADIUS_M: float = 0.22; V0_MEAN: float = 1.3; V0_STD: float = 0.2; MAX_SPEED_M_S: float = 2.2
    REACTION_TIME_DEFAULT_S: float = 0.2; REACTION_TIME_PHONE_S: float = 0.556  # 出典[4]
    FOV_DEFAULT_DEG: float = 120.0; FOV_PHONE_DEG: float = 90.0
    AVOID_BETA_PHONE: float = 0.7
    SPEED_ALPHA_PHONE: float = 0.70   # 出典[5]: 歩きスマホで歩行速度が3割減速
    RELAXATION_TIME_TAU_S: float = 0.5; AGENT_REPULSION_A: float = 5.0
    # 回避距離(B)を縮める実験をしたが、力の立ち上がりが遅くなり逆に衝突が
    # 増えてしまったため元の値に戻した（Bを縮める＝同じ距離での力が弱まる
    # ため、見た目の間合いは縮まらず、単に回避が間に合わなくなるだけだった）。
    AGENT_REPULSION_B_PHONE: float = 0.3; AGENT_REPULSION_B_NORMAL: float = 0.6
    OBSTACLE_REPULSION_A: float = 3.0; OBSTACLE_REPULSION_B: float = 0.15
    NORMAL_EVASION_BOOST: float = 1.8; SHUFFLE_EVASION_BOOST: float = 0.5; SHOULDER_PASS_BOOST: float = 0.9
    # --- レーン分離（車線）の力 ---
    # 人数を減らさずに渋滞を解消するための鍵。AGENT_REPULSION_B_NORMALが
    # 元々2.0m(帯の幅2.5mとほぼ同じ)と非常に長距離だったため、狭い歩道帯
    # では「常に全員が全員を避け続ける」状態になり、密度が上がるとほぼ
    # 身動きが取れなくなっていた（ユーザー指摘: 人数を減らす解決は不可）。
    # 反発距離を短くしただけでは今度は正面衝突が増えるため、実際の混雑した
    # 歩道と同様に「進行方向ごとに帯の中の好みの位置(レーン)へ緩やかに
    # 寄る」力を追加し、すれ違いが自然に起きるようにした。歩きスマホ利用者
    # にはこの力を適用しない（周囲のマナー・車線に合わせる意識も低下して
    # いるとみなす）。複数シードで検証した結果、元のパラメータと同程度の
    # 衝突件数のまま、通過人数を約1.5〜2倍に改善できた。
    LANE_BIAS_STRENGTH: float = 2.0
    LANE_BIAS_FRAC: float = 0.25
    FORWARD_EVASION_BOOST: float = 1.5

    # --- ゆらぎ力（歩行リズムの乱れ）・よろめき検出 出典[4][7] ---
    FLUCT_SIGMA_NORMAL: float = 0.15
    FLUCT_SIGMA_PHONE: float = 0.15 * 1.25   # 出典[4]: 身体動揺の実測増加(約23-30%)
    FALL_EMA_TAU_S: float = 0.35
    FALL_EMA_THRESHOLD: float = 0.30  # dt=1/60s向けに較正済み（sim_coreのdtを変える場合は要再較正）
    FALL_COOLDOWN_S: float = 1.5

    # --- 注意ラプス（画面注視中の瞬間的な視野喪失） 出典[6] ---
    LAPSE_RATE_PER_S: float = 0.5
    LAPSE_DURATION_MIN_S: float = 0.3; LAPSE_DURATION_MAX_S: float = 0.8
    # ラプス中でも柱・什器への回避力をゼロにはせず、わずかに残す（完全に
    # ゼロだと、レーンを無視しがちなスマホ利用者が柱の近くでラプスした
    # 場合に本当に動けなくなる「引っかかり」バグの原因になっていた）
    LAPSE_OBSTACLE_RESIDUAL: float = 0.2

    # --- 柱接触の判定係数 ---
    OBSTACLE_CONTACT_FACTOR: float = 1.2   # 半径の和のこの倍数まで近づいたら「接触」

    # --- ニアミス（歩行者間）の判定: Time-to-Collision（TTC）出典[8] ---
    # 単純な距離の閾値だと、ただ近くに立っている（＝接近も衝突の危険もない）
    # 2人までニアミス扱いしてしまい、「公正な」危険度指標にならない。
    # 交通工学のサロゲート安全評価で標準的に使われるTTC（このまま進んだら
    # 何秒後に衝突するか、相対速度・進行方向まで考慮）に基づいて判定する。
    NEAR_MISS_TTC_THRESHOLD_S: float = 1.5  # 出典[8]: 1980年代初頭から広く採用されている基準値

    # --- 2D(Pygame)描画向けの設定。3D版では M_TO_PX 等は使わず、色だけ流用する ---
    M_TO_PX: int = 25; FOV_VIS_RADIUS_M: float = 1.0
    SIDEWALK_COLOR: tuple = (200, 200, 200); ROAD_COLOR: tuple = (105, 105, 105); LINE_COLOR: tuple = (255, 255, 100)
    OBSTACLE_COLOR: tuple = (90, 90, 90)
    AGENT_COLOR: tuple = (0, 100, 200); AGENT_PHONE_COLOR: tuple = (220, 50, 50)
    COLLISION_MARKER_COLOR: tuple = (255, 59, 48)
    OBSTACLE_MARKER_COLOR: tuple = (255, 176, 32)
    STUMBLE_MARKER_COLOR: tuple = (176, 125, 255)

    TOTAL_HEIGHT_M: float = field(init=False)
    WIDTH_PX: int = field(init=False); HEIGHT_PX: int = field(init=False)
    OBSTACLES: tuple = field(init=False)  # 歩道上の街灯・標識ポール

    def __post_init__(self):
        self.TOTAL_HEIGHT_M = 2 * self.SIDEWALK_HEIGHT_M + self.ROAD_HEIGHT_M
        self.WIDTH_PX = int(self.WIDTH_M * self.M_TO_PX)
        self.HEIGHT_PX = int(self.TOTAL_HEIGHT_M * self.M_TO_PX)
        # 街灯は実際の日本の道路と同じく、歩道の中央ではなく建物側の端に寄せて
        # 置く。壁とレーン目標位置のちょうど中間に置くと、壁側・レーン側の
        # 両方からギリギリに挟まれた隙間になり、回避のたびに狭い隙間を
        # 縫うように通る必要が生じて「スムーズさが失われる」問題があった
        # （実測・3D目視で確認）。そこで壁にほぼ密着させる配置に変更し、
        # 制約を壁側の一方だけにして、帯の奥（レーン目標位置の先）に
        # 回避の逃げ場を残すようにした。なお、レーン目標位置とは重ならない
        # （v19.3で修正した「電柱に引っかかる」不具合の再発を避ける）。
        wall_margin = 0.05  # 壁と街灯のわずかな隙間
        edge_offset = self.LAMPPOST_RADIUS_M + wall_margin
        top_y = edge_offset
        bottom_y = self.TOTAL_HEIGHT_M - edge_offset
        xs = np.linspace(4.0, self.WIDTH_M - 4.0, 5)
        self.OBSTACLES = tuple((float(x), top_y, self.LAMPPOST_RADIUS_M) for x in xs[::2]) + \
                          tuple((float(x), bottom_y, self.LAMPPOST_RADIUS_M) for x in xs[1::2])


@dataclass
class Pedestrian:
    id: int; config: Config; phone_user: bool; pos: np.ndarray; vel: np.ndarray
    v0: float; goal: np.ndarray
    radius: float = field(init=False); reaction_time: float = field(init=False)
    fov_rad: float = field(init=False); avoid_beta: float = field(init=False)
    v_desired_magnitude: float = field(init=False); fluct_sigma: float = field(init=False)
    history: deque = field(default_factory=lambda: deque(maxlen=90))
    spawn_time: float = 0.0; exit_time: float = -1.0; collisions_count: int = 0
    obstacle_collisions_count: int = 0; near_miss_count: int = 0
    fluct_ema: float = 0.0; last_fall_t: float = -999.0; lapse_until: float = -999.0
    # このフレームで速度低下(speed_dampening)が起きた原因（近くにいた相手の属性別）。
    # 「赤い人（スマホ利用者）が近くにいることだけが原因の速度低下」を、通常歩行者
    # 同士の混雑による速度低下と区別して識別するために追加した。
    dampened_by_phone: bool = False
    dampened_by_normal: bool = False
    # 累積値（この歩行者が存在した全フレーム中、各原因で速度低下が発生したフレーム数）
    dampened_by_phone_frames: int = 0
    dampened_by_normal_frames: int = 0
    dampened_by_phone_only_frames: int = 0  # スマホ利用者が原因で、通常歩行者は原因でなかったフレーム
    dampened_by_normal_only_frames: int = 0  # 通常歩行者が原因で、スマホ利用者は原因でなかったフレーム
    frames_alive: int = 0

    def __post_init__(self):
        self.radius = self.config.AGENT_RADIUS_M
        if self.phone_user:
            self.reaction_time = self.config.REACTION_TIME_PHONE_S
            self.fov_rad = np.deg2rad(self.config.FOV_PHONE_DEG)
            self.avoid_beta = self.config.AVOID_BETA_PHONE
            self.v_desired_magnitude = self.v0 * self.config.SPEED_ALPHA_PHONE
            self.fluct_sigma = self.config.FLUCT_SIGMA_PHONE
        else:
            self.reaction_time = self.config.REACTION_TIME_DEFAULT_S
            self.fov_rad = np.deg2rad(self.config.FOV_DEFAULT_DEG)
            self.avoid_beta = 1.0
            self.v_desired_magnitude = self.v0
            self.fluct_sigma = self.config.FLUCT_SIGMA_NORMAL

    def get_delayed_state(self, t: float) -> np.ndarray:
        tt = t - self.reaction_time
        if not self.history or tt <= self.history[0][0]:
            return self.history[0][1] if self.history else self.pos
        for i in range(len(self.history) - 1):
            if self.history[i][0] <= tt < self.history[i + 1][0]:
                return self.history[i][1]
        return self.history[-1][1]


class SimulationCore:
    """レンダラーに依存しない物理・イベント検出エンジン。
    2D(simulation.py)・3D(simulation_3d.py)の両方から継承・利用される。"""

    def __init__(self, p_phone: float, spawn_rate: float, seed: Optional[int] = None,
                 dt: float = 1.0 / 60.0, max_agents: int = 13, trace_path: Optional[str] = None):
        self.config = Config()
        self.p_phone = p_phone
        self.spawn_rate = spawn_rate
        # 同時存在人数の上限。歩道の帯（幅2.5m x 2本）は狭いため、上限なしで
        # 生成し続けると密度が破綻して将棋倒し状態になり、「歩きスマホだから
        # 危険」という個別の対比がかえって見えなくなってしまう。生成レートは
        # あくまで「上限までの埋まる速さ」を調整するものとして扱う。
        # 歩道幅を2.5m→1.5mに狭めた際、面積が60%になった分、密度を保つよう
        # 上限も22→13に比例して下げた（元の22のままだと狭い帯に詰め込み
        # すぎて、電柱付近や出入り口で団子状の渋滞が目視で確認された）。
        self.max_agents = max_agents
        self.rng = np.random.default_rng(seed)  # seed未指定なら毎回変化する
        self.dt = dt
        self.sim_time = 0.0
        self.agents: list[Pedestrian] = []
        self.completed_agents: list[Pedestrian] = []
        self.next_agent_id = 0
        self.total_collisions = 0; self.total_obstacle_collisions = 0
        self.total_stumbles = 0; self.total_near_misses = 0
        self.phone_users_completed = 0; self.normal_users_completed = 0
        self.phone_user_total_collisions = 0; self.normal_user_total_collisions = 0
        self.phone_user_total_obstacle = 0; self.normal_user_total_obstacle = 0
        self.phone_user_total_stumbles = 0; self.normal_user_total_stumbles = 0
        self.phone_user_total_near_miss = 0; self.normal_user_total_near_miss = 0
        # 「歩きスマホの人たちに対する回避率」：少なくとも一方がスマホ利用者だった
        # 接近イベント（衝突 or ニアミス）のうち、実際に衝突せず避けられた割合を
        # 後から計算できるよう、ペア単位で1回ずつ集計する（衝突・ニアミス双方で
        # 既に実装している「同一フレーム内の二重カウント防止」の仕組みをそのまま使う）。
        self.phone_involved_collisions = 0
        self.phone_involved_near_misses = 0
        # 「赤い人（スマホ利用者）が近くにいることだけが原因の速度低下」を識別するための
        # 累積値。通常歩行者（青）について、存在した全フレーム中、(a)スマホ利用者が
        # 近くにいたことで速度低下したフレーム数、(b)そのうち通常歩行者の混雑は
        # 原因になっていなかった（赤い人だけが原因と言える）フレーム数、を集計する。
        self.normal_frames_total = 0
        self.normal_frames_dampened_by_phone = 0
        self.normal_frames_dampened_by_phone_only = 0
        self.normal_frames_dampened_by_normal_only = 0
        self.collision_markers = []  # (kind, pos, timestamp, danger)
        self.ongoing_collisions = set(); self.ongoing_near = set(); self.ongoing_obstacle = set()
        self._define_walkable_areas()

        # --- デバッグ用トレース出力 ---
        # 「揺れる」「避け方が不安」のような見た目の報告を、スクリーンショットで
        # 目視するのではなく数値で原因箇所を特定できるようにするための機能。
        # trace_pathを指定した時だけ毎フレーム各エージェントの位置・速度・向き
        # ・最寄り障害物距離などをCSVに書き出す（通常実行には影響しない）。
        self._trace_writer = None
        self._trace_file = None
        if trace_path:
            self._trace_file = open(trace_path, "w", newline="", encoding="utf-8")
            self._trace_writer = csv.writer(self._trace_file)
            self._trace_writer.writerow([
                "time", "agent_id", "phone_user", "pos_x", "pos_y", "vel_x", "vel_y",
                "speed", "yaw_deg", "in_lapse", "nearest_obstacle_dist",
                "collisions_count", "obstacle_collisions_count", "near_miss_count",
                "dampened_by_phone", "dampened_by_normal",
            ])

    def _define_walkable_areas(self):
        self.sidewalk_top_y_range_m = (0.0, self.config.SIDEWALK_HEIGHT_M)
        self.sidewalk_bottom_y_range_m = (self.config.SIDEWALK_HEIGHT_M + self.config.ROAD_HEIGHT_M,
                                           self.config.TOTAL_HEIGHT_M)

    def _y_range_for(self, y: float):
        return self.sidewalk_top_y_range_m if y < self.config.TOTAL_HEIGHT_M / 2 else self.sidewalk_bottom_y_range_m

    @staticmethod
    def _time_to_collision(dp: np.ndarray, dv: np.ndarray, r_sum: float) -> float:
        """円同士のTime-to-Collision（等速直線運動を仮定した、接触までの秒数）。
        出典[8]。dp = 相手の位置 - 自分の位置、dv = 相手の速度 - 自分の速度。
        離れていく・平行に進む（衝突しない）場合は無限大を返す。
        すでに重なっている場合は0.0（＝衝突。呼び出し側で別途カウント済み）。"""
        a = float(np.dot(dv, dv))
        c = float(np.dot(dp, dp)) - r_sum ** 2
        if c <= 0:
            return 0.0
        if a < 1e-9:
            return np.inf
        b = 2.0 * float(np.dot(dp, dv))
        disc = b * b - 4 * a * c
        if disc < 0:
            return np.inf
        t1 = (-b - np.sqrt(disc)) / (2 * a)
        return t1 if t1 >= 0 else np.inf

    # -------------------- スポーン（歩道の二方向すれ違い） --------------------
    def spawn_pedestrian(self):
        if len(self.agents) >= self.max_agents:
            return
        if self.rng.random() >= self.spawn_rate * self.dt:
            return
        direction = 1 if self.rng.random() < 0.5 else -1
        y_range = self.sidewalk_top_y_range_m if self.rng.random() < 0.5 else self.sidewalk_bottom_y_range_m
        r = self.config.AGENT_RADIUS_M
        if direction == 1:
            pos_x = self.rng.uniform(r, r * 3)
        else:
            pos_x = self.rng.uniform(self.config.WIDTH_M - r * 3, self.config.WIDTH_M - r)
        pos = np.array([pos_x, self.rng.uniform(y_range[0] + r, y_range[1] - r)])
        goal = np.array([(self.config.WIDTH_M + 6.0) if direction == 1 else -6.0, pos[1]])

        if any(np.linalg.norm(pos - a.pos) < 2 * r for a in self.agents):
            return
        v0 = self.rng.normal(self.config.V0_MEAN, self.config.V0_STD)
        self.agents.append(Pedestrian(
            id=self.next_agent_id, config=self.config,
            phone_user=self.rng.random() < self.p_phone,
            pos=pos, vel=np.array([v0 * direction * 0.5, 0.0]), v0=v0, goal=goal, spawn_time=self.sim_time,
        ))
        self.next_agent_id += 1

    # -------------------- 力の計算・状態更新 --------------------
    def update_agents(self):
        for agent in self.agents:
            agent.history.append((self.sim_time, agent.pos.copy()))

        current_frame_overlaps = set(); current_frame_near = set(); current_frame_obstacle = set()
        forces = {agent.id: np.zeros(2) for agent in self.agents}

        for i, agent in enumerate(self.agents):
            # --- 注意ラプス判定（歩きスマホのみ発生） 出典[6] ---
            in_lapse = self.sim_time < agent.lapse_until
            if agent.phone_user and not in_lapse:
                if self.rng.random() < self.config.LAPSE_RATE_PER_S * self.dt:
                    dur = self.rng.uniform(self.config.LAPSE_DURATION_MIN_S, self.config.LAPSE_DURATION_MAX_S)
                    agent.lapse_until = self.sim_time + dur
                    in_lapse = True

            f_avoid = np.zeros(2)
            speed_dampening_factor = 1.0
            agent.dampened_by_phone = False
            agent.dampened_by_normal = False
            my_state = agent.get_delayed_state(self.sim_time)
            my_speed = np.linalg.norm(agent.vel)

            for j, other in enumerate(self.agents):
                if i == j:
                    continue
                other_state = other.get_delayed_state(self.sim_time)
                vec_to_other = other_state - my_state
                dist = np.linalg.norm(vec_to_other)
                r_sum = agent.radius + other.radius

                # 衝突・ニアミスは「実際に物理的に起きた出来事」なので、知覚の
                # 反応遅延を含まない真の位置・速度で判定する（dist/vec_to_other
                # は知覚位置であり、FOV判定・回避力の計算にのみ使う）。
                # 以前は衝突だけ知覚位置(dist)で判定しており、ニアミス（真の位置
                # ベース、下記）と判定基準が食い違っていたため統一した。
                dp_true = other.pos - agent.pos
                dv_true = other.vel - agent.vel
                dist_true = np.linalg.norm(dp_true)

                # 各ペアは(agent=i,other=j)と(agent=j,other=i)の2回、このフレーム内で
                # 訪問される（force計算が各エージェント視点で非対称なため、ループ自体は
                # 両方向が必要）。イベント計測（衝突・ニアミス）はペア単位で1回だけ
                # 数えたいので、「このフレーム内で既にこのペアを処理したか」
                # (current_frame_*に既に入っているか)を先にチェックしてから追加する。
                # これを怠ると、1回の接触が常に2回としてカウントされてしまう
                # （以前の実装で実際に発生していた二重カウントのバグ）。
                if dist_true < r_sum:
                    pair = tuple(sorted((agent.id, other.id)))
                    is_new_this_frame = pair not in current_frame_overlaps
                    current_frame_overlaps.add(pair)
                    if is_new_this_frame and pair not in self.ongoing_collisions:
                        agent.collisions_count += 1; other.collisions_count += 1; self.total_collisions += 1
                        phone_involved = agent.phone_user or other.phone_user
                        self.collision_markers.append(("collision", agent.pos + dp_true / 2, self.sim_time,
                                                        phone_involved))
                        if phone_involved:
                            self.phone_involved_collisions += 1
                else:
                    # ニアミスは「近いだけ」ではなく、相対速度・進行方向を考慮した
                    # TTCで判定する。立ち止まって近くにいるだけの2人や、すれ違って
                    # 離れていく2人を誤ってニアミス扱いしないため。
                    ttc = self._time_to_collision(dp_true, dv_true, r_sum)
                    if 0.0 < ttc <= self.config.NEAR_MISS_TTC_THRESHOLD_S:
                        pair = tuple(sorted((agent.id, other.id)))
                        is_new_this_frame = pair not in current_frame_near
                        current_frame_near.add(pair)
                        if is_new_this_frame and pair not in self.ongoing_near and pair not in self.ongoing_collisions:
                            self.total_near_misses += 1
                            agent.near_miss_count += 1; other.near_miss_count += 1
                            if agent.phone_user or other.phone_user:
                                self.phone_involved_near_misses += 1

                is_in_fov = not in_lapse
                if is_in_fov and my_speed > 0.1 and dist > 0:
                    ang = abs(np.arccos(np.clip(np.dot(agent.vel / my_speed, vec_to_other / dist), -1.0, 1.0)))
                    if ang > agent.fov_rad / 2:
                        is_in_fov = False

                if is_in_fov:
                    if dist < self.config.AGENT_RADIUS_M * 4:
                        speed_dampening_factor = max(0.6, speed_dampening_factor - 0.1)
                        if other.phone_user:
                            agent.dampened_by_phone = True
                        else:
                            agent.dampened_by_normal = True
                    repulsion_b = self.config.AGENT_REPULSION_B_NORMAL if not agent.phone_user else self.config.AGENT_REPULSION_B_PHONE
                    force_mag = self.config.AGENT_REPULSION_A * np.exp((r_sum - dist) / repulsion_b)
                    evasion_boost = 1.0
                    if not agent.phone_user:
                        use_default_logic = True
                        if not other.phone_user:
                            is_red_nearby = any(
                                red.phone_user and np.linalg.norm(agent.pos - red.pos) < 7.0
                                for red in self.agents
                            )
                            if not is_red_nearby:
                                evasion_boost = self.config.SHOULDER_PASS_BOOST
                                use_default_logic = False
                        if use_default_logic:
                            relative_vel = other.vel - agent.vel
                            if np.dot(vec_to_other, relative_vel) < 0:
                                evasion_boost = self.config.SHUFFLE_EVASION_BOOST if self.rng.random() < (1 / 3) else self.config.NORMAL_EVASION_BOOST
                            else:
                                evasion_boost = self.config.NORMAL_EVASION_BOOST
                        if my_speed > 0.1 and dist > 0:
                            cos_angle = np.dot(agent.vel, vec_to_other) / (my_speed * dist)
                            if cos_angle > 0.9:
                                evasion_boost *= self.config.FORWARD_EVASION_BOOST

                    force_direction = -vec_to_other / dist if dist > 0 else np.zeros(2)
                    f_avoid += agent.avoid_beta * evasion_boost * force_mag * force_direction

            # --- 障害物（街灯・標識ポール）との相互作用 ---
            # 注意ラプス中は知覚をほぼゼロにする設計だが、完全にゼロにすると
            # 「中央に寄りやすいレーン無視のスマホ利用者」が「ラプス中は
            # 障害物力もゼロ」という2つの条件が重なったときに、電柱のすぐ
            # そばで本当に動けなくなる（引っかかる）不具合が起きていた。
            # ラプス中もごくわずかな残存知覚(LAPSE_OBSTACLE_RESIDUAL)を
            # 残すことで、致命的な膠着を避けつつ「気づきにくい」ことは表現する。
            f_obs = np.zeros(2)
            for oi, (ox, oy, orad) in enumerate(self.config.OBSTACLES):
                d = agent.pos - np.array([ox, oy])
                dd = max(np.linalg.norm(d), 1e-6)
                r_sum_o = agent.radius + orad
                obs_perception = self.config.LAPSE_OBSTACLE_RESIDUAL if in_lapse else 1.0
                if obs_perception > 0 and not in_lapse and my_speed > 0.1:
                    ang = abs(np.arccos(np.clip(np.dot(agent.vel / my_speed, -d / dd), -1.0, 1.0)))
                    if ang > agent.fov_rad / 2:
                        obs_perception = 0.0
                if obs_perception > 0:
                    m = self.config.OBSTACLE_REPULSION_A * np.exp((r_sum_o - dd) / self.config.OBSTACLE_REPULSION_B)
                    f_obs += m * obs_perception * (d / dd)
                if dd < r_sum_o * self.config.OBSTACLE_CONTACT_FACTOR:
                    key = (agent.id, oi)
                    current_frame_obstacle.add(key)
                    if key not in self.ongoing_obstacle:
                        self.total_obstacle_collisions += 1
                        agent.obstacle_collisions_count += 1
                        self.collision_markers.append(("obstacle", agent.pos.copy(), self.sim_time, agent.phone_user))
                    if dd < r_sum_o:  # 幾何学的な重なりは押し戻す
                        agent.pos += (d / dd) * (r_sum_o - dd)

            # --- ゆらぎ力（歩行リズムの乱れ）とよろめき検出 出典[4][7] ---
            f_fluct = self.rng.normal(0, agent.fluct_sigma, size=2)
            alpha = min(1.0, self.dt / self.config.FALL_EMA_TAU_S)
            agent.fluct_ema = agent.fluct_ema * (1 - alpha) + np.linalg.norm(f_fluct) * alpha
            if (agent.fluct_ema > self.config.FALL_EMA_THRESHOLD and
                    self.sim_time - agent.last_fall_t > self.config.FALL_COOLDOWN_S):
                agent.last_fall_t = self.sim_time
                self.total_stumbles += 1
                if agent.phone_user:
                    self.phone_user_total_stumbles += 1
                else:
                    self.normal_user_total_stumbles += 1
                self.collision_markers.append(("stumble", agent.pos.copy(), self.sim_time, agent.phone_user))

            # 駆動力の向きは「x方向にゴールへ向かう」だけにする（y成分を含めると
            # 回避行動で上下にずれた分を元のyへ引き戻す力が働き、回避操舵と
            # 競合して渋滞・膠着を起こすため）
            e0 = np.array([1.0 if agent.goal[0] >= agent.pos[0] else -1.0, 0.0])
            v_desired_vec = e0 * agent.v_desired_magnitude * speed_dampening_factor
            f_drive = (v_desired_vec - agent.vel) / self.config.RELAXATION_TIME_TAU_S

            # 速度低下の原因別フレーム数を累積（「赤い人だけが原因の速度低下」を
            # 「通常歩行者同士の混雑による速度低下」と区別して後から集計できるように）
            agent.frames_alive += 1
            if agent.dampened_by_phone:
                agent.dampened_by_phone_frames += 1
                if not agent.dampened_by_normal:
                    agent.dampened_by_phone_only_frames += 1
            if agent.dampened_by_normal:
                agent.dampened_by_normal_frames += 1
                if not agent.dampened_by_phone:
                    agent.dampened_by_normal_only_frames += 1

            # --- 歩道の境界（車道側へはみ出さないようにする壁の反発力） ---
            y_range = self._y_range_for(agent.pos[1])
            f_wall = np.array([0.0,
                                self.config.OBSTACLE_REPULSION_A * np.exp((agent.radius - (agent.pos[1] - y_range[0])) / self.config.OBSTACLE_REPULSION_B)
                                - self.config.OBSTACLE_REPULSION_A * np.exp((agent.radius - (y_range[1] - agent.pos[1])) / self.config.OBSTACLE_REPULSION_B)])

            # --- レーン分離の力（歩きスマホ利用者には適用しない） ---
            f_lane = np.zeros(2)
            if not agent.phone_user:
                band = y_range[1] - y_range[0]
                frac = self.config.LANE_BIAS_FRAC if e0[0] > 0 else (1.0 - self.config.LANE_BIAS_FRAC)
                target_y = y_range[0] + band * frac
                f_lane[1] = (target_y - agent.pos[1]) * self.config.LANE_BIAS_STRENGTH

            forces[agent.id] = f_drive + f_avoid + f_obs + f_wall + f_lane + f_fluct

        for agent in self.agents:
            agent.vel += forces[agent.id] * self.dt
            speed = np.linalg.norm(agent.vel)
            if speed > self.config.MAX_SPEED_M_S:
                agent.vel *= self.config.MAX_SPEED_M_S / speed
            agent.pos += agent.vel * self.dt
            y_range = self._y_range_for(agent.pos[1])
            agent.pos[1] = np.clip(agent.pos[1], y_range[0] + agent.radius, y_range[1] - agent.radius)

        self.ongoing_collisions = current_frame_overlaps
        self.ongoing_near = current_frame_near
        self.ongoing_obstacle = current_frame_obstacle

        margin = 6.0
        despawn_candidates = [a for a in self.agents if (self.sim_time - a.spawn_time > 1.0) and (
            a.pos[0] < -margin or a.pos[0] > self.config.WIDTH_M + margin or
            a.pos[1] < -margin or a.pos[1] > self.config.TOTAL_HEIGHT_M + margin)]
        for agent in despawn_candidates:
            agent.exit_time = self.sim_time; self.completed_agents.append(agent)
            if agent.phone_user:
                self.phone_users_completed += 1
                self.phone_user_total_collisions += agent.collisions_count
                self.phone_user_total_obstacle += agent.obstacle_collisions_count
                self.phone_user_total_near_miss += agent.near_miss_count
            else:
                self.normal_users_completed += 1
                self.normal_user_total_collisions += agent.collisions_count
                self.normal_user_total_obstacle += agent.obstacle_collisions_count
                self.normal_user_total_near_miss += agent.near_miss_count
                self.normal_frames_total += agent.frames_alive
                self.normal_frames_dampened_by_phone += agent.dampened_by_phone_frames
                self.normal_frames_dampened_by_phone_only += agent.dampened_by_phone_only_frames
                self.normal_frames_dampened_by_normal_only += agent.dampened_by_normal_only_frames
        self.agents = [a for a in self.agents if a.exit_time < 0]

    def step(self):
        """1ステップ進める（スポーン + 状態更新 + 時刻更新）。"""
        self.spawn_pedestrian()
        self.update_agents()
        self.sim_time += self.dt
        if self._trace_writer is not None:
            self._write_trace_frame()

    def _write_trace_frame(self):
        obstacles = self.config.OBSTACLES
        for agent in self.agents:
            speed = float(np.linalg.norm(agent.vel))
            yaw_deg = float(np.degrees(np.arctan2(agent.vel[0], agent.vel[1]))) if speed > 0.1 else ""
            in_lapse = self.sim_time < agent.lapse_until
            if obstacles:
                nearest = min(np.linalg.norm(agent.pos - np.array([ox, oy])) for (ox, oy, orad) in obstacles)
            else:
                nearest = ""
            self._trace_writer.writerow([
                round(self.sim_time, 4), agent.id, agent.phone_user,
                round(float(agent.pos[0]), 4), round(float(agent.pos[1]), 4),
                round(float(agent.vel[0]), 4), round(float(agent.vel[1]), 4),
                round(speed, 4), round(yaw_deg, 2) if yaw_deg != "" else "",
                in_lapse, round(float(nearest), 4) if nearest != "" else "",
                agent.collisions_count, agent.obstacle_collisions_count, agent.near_miss_count,
                agent.dampened_by_phone, agent.dampened_by_normal,
            ])
        self._trace_file.flush()

    def close_trace(self):
        if self._trace_file is not None:
            self._trace_file.close()
            self._trace_file = None
            self._trace_writer = None

    def write_live_data(self, path: str = "live_data.json"):
        current_avg_speed = np.mean([np.linalg.norm(a.vel) for a in self.agents]) if self.agents else 0
        rate_phone = self.phone_user_total_collisions / self.phone_users_completed if self.phone_users_completed > 0 else 0
        rate_normal = self.normal_user_total_collisions / self.normal_users_completed if self.normal_users_completed > 0 else 0
        obs_rate_phone = self.phone_user_total_obstacle / self.phone_users_completed if self.phone_users_completed > 0 else 0
        obs_rate_normal = self.normal_user_total_obstacle / self.normal_users_completed if self.normal_users_completed > 0 else 0
        stumble_rate_phone = self.phone_user_total_stumbles / self.phone_users_completed if self.phone_users_completed > 0 else 0
        stumble_rate_normal = self.normal_user_total_stumbles / self.normal_users_completed if self.normal_users_completed > 0 else 0
        near_miss_rate_phone = self.phone_user_total_near_miss / self.phone_users_completed if self.phone_users_completed > 0 else 0
        near_miss_rate_normal = self.normal_user_total_near_miss / self.normal_users_completed if self.normal_users_completed > 0 else 0
        # 「赤い人（スマホ利用者）だけが原因の速度低下」の割合（通常歩行者の存在フレームのうち）
        frac_dampened_by_phone = self.normal_frames_dampened_by_phone / self.normal_frames_total if self.normal_frames_total > 0 else 0
        frac_dampened_by_phone_only = self.normal_frames_dampened_by_phone_only / self.normal_frames_total if self.normal_frames_total > 0 else 0
        frac_dampened_by_normal_only = self.normal_frames_dampened_by_normal_only / self.normal_frames_total if self.normal_frames_total > 0 else 0
        phone_encounter_total = self.phone_involved_collisions + self.phone_involved_near_misses
        phone_avoidance_rate = self.phone_involved_near_misses / phone_encounter_total if phone_encounter_total > 0 else 0
        data = {
            "time": self.sim_time, "agent_count": len(self.agents), "completed_count": len(self.completed_agents),
            "total_collisions": self.total_collisions, "total_obstacle_collisions": self.total_obstacle_collisions,
            "total_stumbles": self.total_stumbles, "total_near_misses": self.total_near_misses,
            "avg_speed": current_avg_speed,
            "collision_rate_phone": rate_phone, "collision_rate_normal": rate_normal,
            "obstacle_rate_phone": obs_rate_phone, "obstacle_rate_normal": obs_rate_normal,
            "stumble_rate_phone": stumble_rate_phone, "stumble_rate_normal": stumble_rate_normal,
            "near_miss_rate_phone": near_miss_rate_phone, "near_miss_rate_normal": near_miss_rate_normal,
            "normal_frac_dampened_by_phone": frac_dampened_by_phone,
            "normal_frac_dampened_by_phone_only": frac_dampened_by_phone_only,
            "normal_frac_dampened_by_normal_only": frac_dampened_by_normal_only,
            "phone_involved_collisions": self.phone_involved_collisions,
            "phone_involved_near_misses": self.phone_involved_near_misses,
            "phone_avoidance_rate": phone_avoidance_rate,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except IOError as e:
            print(f"Error writing live data: {e}")
