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
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


@dataclass
class Config:
    # --- 歩道空間の定義（二方向のすれ違い） ---
    WIDTH_M: float = 30.0; SIDEWALK_HEIGHT_M: float = 2.5; ROAD_HEIGHT_M: float = 4.0
    LAMPPOST_RADIUS_M: float = 0.15  # 街灯・標識ポール（障害物）の太さ

    AGENT_RADIUS_M: float = 0.3; V0_MEAN: float = 1.3; V0_STD: float = 0.2; MAX_SPEED_M_S: float = 2.2
    REACTION_TIME_DEFAULT_S: float = 0.2; REACTION_TIME_PHONE_S: float = 0.556  # 出典[4]
    FOV_DEFAULT_DEG: float = 120.0; FOV_PHONE_DEG: float = 90.0
    AVOID_BETA_PHONE: float = 0.7
    SPEED_ALPHA_PHONE: float = 0.70   # 出典[5]: 歩きスマホで歩行速度が3割減速
    RELAXATION_TIME_TAU_S: float = 0.5; AGENT_REPULSION_A: float = 2.1
    AGENT_REPULSION_B_PHONE: float = 0.3; AGENT_REPULSION_B_NORMAL: float = 2.0
    OBSTACLE_REPULSION_A: float = 3.0; OBSTACLE_REPULSION_B: float = 0.15
    NORMAL_EVASION_BOOST: float = 1.8; SHUFFLE_EVASION_BOOST: float = 0.5; SHOULDER_PASS_BOOST: float = 0.9
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

    # --- 柱接触／ニアミスの判定係数 ---
    OBSTACLE_CONTACT_FACTOR: float = 1.4   # 半径の和のこの倍数まで近づいたら「接触」
    NEAR_MISS_FACTOR: float = 1.6          # 半径の和のこの倍数まで近づいたら「ニアミス」

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
        top_y = self.SIDEWALK_HEIGHT_M * 0.25
        bottom_y = self.SIDEWALK_HEIGHT_M + self.ROAD_HEIGHT_M + self.SIDEWALK_HEIGHT_M * 0.75
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
    obstacle_collisions_count: int = 0
    fluct_ema: float = 0.0; last_fall_t: float = -999.0; lapse_until: float = -999.0

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
                 dt: float = 1.0 / 60.0):
        self.config = Config()
        self.p_phone = p_phone
        self.spawn_rate = spawn_rate
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
        self.collision_markers = []  # (kind, pos, timestamp, danger)
        self.ongoing_collisions = set(); self.ongoing_near = set(); self.ongoing_obstacle = set()
        self._define_walkable_areas()

    def _define_walkable_areas(self):
        self.sidewalk_top_y_range_m = (0.0, self.config.SIDEWALK_HEIGHT_M)
        self.sidewalk_bottom_y_range_m = (self.config.SIDEWALK_HEIGHT_M + self.config.ROAD_HEIGHT_M,
                                           self.config.TOTAL_HEIGHT_M)

    def _y_range_for(self, y: float):
        return self.sidewalk_top_y_range_m if y < self.config.TOTAL_HEIGHT_M / 2 else self.sidewalk_bottom_y_range_m

    # -------------------- スポーン（歩道の二方向すれ違い） --------------------
    def spawn_pedestrian(self):
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
            my_state = agent.get_delayed_state(self.sim_time)
            my_speed = np.linalg.norm(agent.vel)

            for j, other in enumerate(self.agents):
                if i == j:
                    continue
                other_state = other.get_delayed_state(self.sim_time)
                vec_to_other = other_state - my_state
                dist = np.linalg.norm(vec_to_other)
                r_sum = agent.radius + other.radius

                if dist < r_sum:
                    pair = tuple(sorted((agent.id, other.id)))
                    current_frame_overlaps.add(pair)
                    if pair not in self.ongoing_collisions:
                        agent.collisions_count += 1; other.collisions_count += 1; self.total_collisions += 1
                        self.collision_markers.append(("collision", agent.pos + vec_to_other / 2, self.sim_time,
                                                        agent.phone_user or other.phone_user))
                elif dist < r_sum * self.config.NEAR_MISS_FACTOR:
                    pair = tuple(sorted((agent.id, other.id)))
                    current_frame_near.add(pair)
                    if pair not in self.ongoing_near and pair not in self.ongoing_collisions:
                        self.total_near_misses += 1

                is_in_fov = not in_lapse
                if is_in_fov and my_speed > 0.1 and dist > 0:
                    ang = abs(np.arccos(np.clip(np.dot(agent.vel / my_speed, vec_to_other / dist), -1.0, 1.0)))
                    if ang > agent.fov_rad / 2:
                        is_in_fov = False

                if is_in_fov:
                    if dist < self.config.AGENT_RADIUS_M * 4:
                        speed_dampening_factor = max(0.6, speed_dampening_factor - 0.1)
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
            f_obs = np.zeros(2)
            for oi, (ox, oy, orad) in enumerate(self.config.OBSTACLES):
                d = agent.pos - np.array([ox, oy])
                dd = max(np.linalg.norm(d), 1e-6)
                r_sum_o = agent.radius + orad
                obs_in_fov = not in_lapse
                if obs_in_fov and my_speed > 0.1:
                    ang = abs(np.arccos(np.clip(np.dot(agent.vel / my_speed, -d / dd), -1.0, 1.0)))
                    if ang > agent.fov_rad / 2:
                        obs_in_fov = False
                if obs_in_fov:
                    m = self.config.OBSTACLE_REPULSION_A * np.exp((r_sum_o - dd) / self.config.OBSTACLE_REPULSION_B)
                    f_obs += m * (d / dd)
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

            # --- 歩道の境界（車道側へはみ出さないようにする壁の反発力） ---
            y_range = self._y_range_for(agent.pos[1])
            f_wall = np.array([0.0,
                                self.config.OBSTACLE_REPULSION_A * np.exp((agent.radius - (agent.pos[1] - y_range[0])) / self.config.OBSTACLE_REPULSION_B)
                                - self.config.OBSTACLE_REPULSION_A * np.exp((agent.radius - (y_range[1] - agent.pos[1])) / self.config.OBSTACLE_REPULSION_B)])

            forces[agent.id] = f_drive + f_avoid + f_obs + f_wall + f_fluct

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
            else:
                self.normal_users_completed += 1
                self.normal_user_total_collisions += agent.collisions_count
                self.normal_user_total_obstacle += agent.obstacle_collisions_count
        self.agents = [a for a in self.agents if a.exit_time < 0]

    def step(self):
        """1ステップ進める（スポーン + 状態更新 + 時刻更新）。"""
        self.spawn_pedestrian()
        self.update_agents()
        self.sim_time += self.dt

    def write_live_data(self, path: str = "live_data.json"):
        current_avg_speed = np.mean([np.linalg.norm(a.vel) for a in self.agents]) if self.agents else 0
        rate_phone = self.phone_user_total_collisions / self.phone_users_completed if self.phone_users_completed > 0 else 0
        rate_normal = self.normal_user_total_collisions / self.normal_users_completed if self.normal_users_completed > 0 else 0
        obs_rate_phone = self.phone_user_total_obstacle / self.phone_users_completed if self.phone_users_completed > 0 else 0
        obs_rate_normal = self.normal_user_total_obstacle / self.normal_users_completed if self.normal_users_completed > 0 else 0
        stumble_rate_phone = self.phone_user_total_stumbles / self.phone_users_completed if self.phone_users_completed > 0 else 0
        stumble_rate_normal = self.normal_user_total_stumbles / self.normal_users_completed if self.normal_users_completed > 0 else 0
        data = {
            "time": self.sim_time, "agent_count": len(self.agents), "completed_count": len(self.completed_agents),
            "total_collisions": self.total_collisions, "total_obstacle_collisions": self.total_obstacle_collisions,
            "total_stumbles": self.total_stumbles, "total_near_misses": self.total_near_misses,
            "avg_speed": current_avg_speed,
            "collision_rate_phone": rate_phone, "collision_rate_normal": rate_normal,
            "obstacle_rate_phone": obs_rate_phone, "obstacle_rate_normal": obs_rate_normal,
            "stumble_rate_phone": stumble_rate_phone, "stumble_rate_normal": stumble_rate_normal,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except IOError as e:
            print(f"Error writing live data: {e}")
