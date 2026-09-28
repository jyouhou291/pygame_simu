# =============================================================================
# 歩きスマホ行動シミュレーション (v17.0 - Simulation Engine)
#
# v17.0 での変更点（駅コンコース版への改訂）:
#   - 舞台を「歩道+車道」から「駅コンコース」に変更（改札12ヶ所、柱・キオス
#     クの障害物を追加、4方向（改札⇔ホーム縦方向／売店・別ホームへの横方向）
#     の人流に対応）
#   - 柱・什器への接触を独立イベントとして検出（歩きスマホの典型的な事故）
#   - 「よろめき（転倒リスク）」を、他者との衝突や回避操舵とは無関係な
#     ゆらぎ力（歩行リズムの乱れ）の指数移動平均から検出
#   - 「注意ラプス」（画面に視線が固定され、前方も含め周囲を一切見ていない
#     瞬間）を追加。FOV角度による緩やかな視野制限だけでは、正面の柱には
#     常に気づいてしまうため
#   - ニアミス（衝突には至らないが接近した回数）を追加
#   - 毎回同じ乱数シード(42)で固定されていたため実行結果が常に同一だった
#     問題を修正し、--seed 未指定時はランダムに変化するようにした
#   - パラメータの一部を実測論文の値に合わせて補正（後述）
#
# 参考文献（歩きスマホ関連パラメータの根拠）:
#   [1] 歩きスマホが歩行に及ぼす影響について (J-STAGE)
#       https://www.jstage.jst.go.jp/article/hppt/6/1/6_35/_pdf
#   [2] 歩行中のスマートフォン使用が歩行動作に及ぼす影響 (CiNii)
#   [3] 歩行中の携帯情報端末使用の実態とその行動特性に関する研究 (JSCE)
#   [4] 小松史旺ほか(2015)「歩きスマホが反応時間および歩行動作に与える影響」
#       日本人間工学会第51巻特別号, 1E4-4
#       → 反応時間: 通常速度で約3.8倍、70%速度で約2.3倍に有意増加(p<=0.05)
#       → 身体動揺(最大ブレ幅): 通常速度で約30%、70%速度で約23%増加
#         （本コードの REACTION_TIME_PHONE_S はこの範囲内に収まっている。
#          FLUCT_SIGMA_PHONE はこの実測比率(約1.25倍)に基づき新規に追加）
#   [5] Lamberg, E.M. & Muratori, L.M. (2012). Cellphones change the way we
#       walk. Gait & Posture, 35(4), 688-690.
#       → 歩きスマホで歩行速度が約3割減速（SPEED_ALPHA_PHONEの根拠）
#   [6] ソフトバンクニュース「ながらスマホの危険性を視線計測で検証。視界の
#       “95%”が消える？」https://www.softbank.jp/sbnews/entry/20191101_01
#       → 画面注視中は視界の面積が約1/20になるという視線計測結果。
#         常時ではなく画面を見ている瞬間の効果と考え、本コードでは
#         「注意ラプス」として断続的に再現する
#   [7] 京都大学 研究成果(2024)「歩きスマホによる内因性転倒リスクの増大」
#       https://www.kyoto-u.ac.jp/ja/research-news/2024-07-19
#       → 他者や段差がなくても、歩きスマホは内因的に歩行を不安定にする
#         （「よろめき」検出の根拠）
# =============================================================================
import pygame
import numpy as np
import argparse
import math
import json
from dataclasses import dataclass, field
from collections import deque
import sys
import os

# --- 定数・設定クラス ---
@dataclass
class Config:
    # --- 駅コンコース空間の定義 ---
    WIDTH_M: float = 30.0
    HEIGHT_M: float = 18.0
    GATE_COUNT: int = 12
    OBSTACLES: tuple = ((8.0, 6.0, 0.5), (22.0, 6.0, 0.5), (8.0, 12.5, 0.5),
                         (22.0, 12.5, 0.5), (15.0, 9.0, 1.1))  # 柱4本+中央キオスク

    AGENT_RADIUS_M: float = 0.3; V0_MEAN: float = 1.3; V0_STD: float = 0.2; MAX_SPEED_M_S: float = 2.2
    REACTION_TIME_DEFAULT_S: float = 0.2; REACTION_TIME_PHONE_S: float = 0.556  # 出典[4]
    FOV_DEFAULT_DEG: float = 120.0; FOV_PHONE_DEG: float = 90.0
    AVOID_BETA_PHONE: float = 0.7
    SPEED_ALPHA_PHONE: float = 0.70   # 出典[5]: 歩きスマホで歩行速度が3割減速（旧版の0.85から補正）
    RELAXATION_TIME_TAU_S: float = 0.5; AGENT_REPULSION_A: float = 2.1
    AGENT_REPULSION_B_PHONE: float = 0.3; AGENT_REPULSION_B_NORMAL: float = 2.0
    OBSTACLE_REPULSION_A: float = 3.0; OBSTACLE_REPULSION_B: float = 0.15
    NORMAL_EVASION_BOOST: float = 1.8; SHUFFLE_EVASION_BOOST: float = 0.5; SHOULDER_PASS_BOOST: float = 0.9
    FORWARD_EVASION_BOOST: float = 1.5

    # --- 新規: ゆらぎ力（歩行リズムの乱れ）・よろめき検出 出典[4][7] ---
    FLUCT_SIGMA_NORMAL: float = 0.15
    FLUCT_SIGMA_PHONE: float = 0.15 * 1.25   # 出典[4]: 身体動揺の実測増加(約23-30%)
    FALL_EMA_TAU_S: float = 0.35
    # 注: FALL_EMA_THRESHOLD は EMA の時定数(FALL_EMA_TAU_S)と実際の
    # タイムステップ(dt=1/60s)の組み合わせで適切な値が変わる（dtが細かい
    # ほどEMAが滑らかになり閾値を超えにくくなる）。0.30 は dt=1/60s において
    # 「通常歩行者はほぼ発生せず、歩きスマホ利用者では有意に発生する」水準
    # になるよう、専用の較正スクリプトで再確認した値。
    FALL_EMA_THRESHOLD: float = 0.30
    FALL_COOLDOWN_S: float = 1.5

    # --- 新規: 注意ラプス（画面注視中の瞬間的な視野喪失） 出典[6] ---
    LAPSE_RATE_PER_S: float = 0.5
    LAPSE_DURATION_MIN_S: float = 0.3; LAPSE_DURATION_MAX_S: float = 0.8

    # --- 新規: 柱接触／ニアミスの判定係数 ---
    OBSTACLE_CONTACT_FACTOR: float = 1.4   # 半径の和のこの倍数まで近づいたら「接触」
    NEAR_MISS_FACTOR: float = 1.6          # 半径の和のこの倍数まで近づいたら「ニアミス」

    M_TO_PX: int = 25; FOV_VIS_RADIUS_M: float = 1.0
    FLOOR_COLOR: tuple = (35, 40, 47); FLOOR_LINE_COLOR: tuple = (47, 54, 63)
    GATE_COLOR: tuple = (51, 59, 70); OBSTACLE_COLOR: tuple = (69, 79, 92)
    AGENT_COLOR: tuple = (79, 168, 255); AGENT_PHONE_COLOR: tuple = (255, 77, 77)
    COLLISION_MARKER_COLOR: tuple = (255, 59, 48)
    OBSTACLE_MARKER_COLOR: tuple = (255, 176, 32)
    STUMBLE_MARKER_COLOR: tuple = (176, 125, 255)
    WIDTH_PX: int = field(init=False); HEIGHT_PX: int = field(init=False)
    GATE_XS_M: tuple = field(init=False)

    def __post_init__(self):
        self.WIDTH_PX = int(self.WIDTH_M * self.M_TO_PX)
        self.HEIGHT_PX = int(self.HEIGHT_M * self.M_TO_PX)
        self.GATE_XS_M = tuple(np.linspace(1.5, self.WIDTH_M - 1.5, self.GATE_COUNT))


# --- エージェントクラス ---
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

    def draw(self, s: pygame.Surface, in_lapse: bool):
        px_pos = (int(self.pos[0] * self.config.M_TO_PX), int(self.pos[1] * self.config.M_TO_PX))
        color = self.config.AGENT_PHONE_COLOR if self.phone_user else self.config.AGENT_COLOR
        pygame.draw.circle(s, color, px_pos, int(self.config.AGENT_RADIUS_M * self.config.M_TO_PX))
        if np.linalg.norm(self.vel) > 0.1 and not in_lapse:
            a = math.atan2(-self.vel[1], self.vel[0])
            r = int(self.config.FOV_VIS_RADIUS_M * self.config.M_TO_PX * (0.7 if self.phone_user else 1.3))
            rect = pygame.Rect(px_pos[0] - r, px_pos[1] - r, 2 * r, 2 * r)
            sa, ea = a - self.fov_rad / 2, a + self.fov_rad / 2
            fov_c = tuple(min(255, c + 60) for c in color)
            try:
                pygame.draw.arc(s, fov_c, rect, sa, ea, 1)
                pygame.draw.line(s, fov_c, px_pos, (px_pos[0] + r * math.cos(sa), px_pos[1] - r * math.sin(sa)), 1)
                pygame.draw.line(s, fov_c, px_pos, (px_pos[0] + r * math.cos(ea), px_pos[1] - r * math.sin(ea)), 1)
            except TypeError:
                pass
        elif in_lapse:
            # 注意ラプス中は目印として頭上に小さな赤い点滅マークを出す
            pygame.draw.circle(s, (255, 255, 255), (px_pos[0], px_pos[1] - 14), 2)


# --- シミュレーション本体クラス ---
class Simulation:
    def __init__(self, args):
        self.args = args; self.config = Config()
        self.rng = np.random.default_rng(args.seed)  # 修正: seed未指定なら毎回変化する
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.WIDTH_PX, self.config.HEIGHT_PX))
        pygame.display.set_caption("歩きスマホ行動シミュレーション（駅コンコース）")
        self.font = pygame.font.Font(None, 26); self.big_font = pygame.font.Font(None, 72)
        self.clock = pygame.time.Clock(); self.dt = 1.0 / 60.0
        self.is_running = True; self.sim_time = 0.0; self.is_paused = False
        self.speed_multiplier = args.speed
        self.agents = []; self.completed_agents = []; self.next_agent_id = 0
        self.total_collisions = 0; self.total_obstacle_collisions = 0; self.total_stumbles = 0; self.total_near_misses = 0
        self.phone_users_completed = 0; self.normal_users_completed = 0
        self.phone_user_total_collisions = 0; self.normal_user_total_collisions = 0
        self.phone_user_total_obstacle = 0; self.normal_user_total_obstacle = 0
        self.phone_user_total_stumbles = 0; self.normal_user_total_stumbles = 0
        self.collision_markers = []
        self.ongoing_collisions = set(); self.ongoing_near = set(); self.ongoing_obstacle = set()

    # -------------------- スポーン（駅コンコースの4方向流動） --------------------
    def _spawn_pedestrian(self):
        if self.rng.random() >= self.args.spawn_rate * self.dt:
            return
        flow = self.rng.choice(["down", "up", "left", "right"], p=[0.37, 0.37, 0.13, 0.13])
        gx = float(self.rng.choice(self.config.GATE_XS_M))
        W, H = self.config.WIDTH_M, self.config.HEIGHT_M
        if flow == "down":
            pos = np.array([gx + self.rng.uniform(-0.5, 0.5), -1.0])
            goal = np.array([gx + self.rng.uniform(-1, 1), H + 6])
        elif flow == "up":
            pos = np.array([gx + self.rng.uniform(-0.5, 0.5), H + 1.0])
            goal = np.array([gx, -6.0])
        elif flow == "right":
            pos = np.array([-1.0, self.rng.uniform(2.0, H - 2.0)])
            goal = np.array([W + 6, pos[1] + self.rng.uniform(-1.5, 1.5)])
        else:
            pos = np.array([W + 1.0, self.rng.uniform(2.0, H - 2.0)])
            goal = np.array([-6.0, pos[1] + self.rng.uniform(-1.5, 1.5)])

        if any(np.linalg.norm(pos - a.pos) < 2 * self.config.AGENT_RADIUS_M for a in self.agents):
            return
        v0 = self.rng.normal(self.config.V0_MEAN, self.config.V0_STD)
        heading = (goal - pos) / max(np.linalg.norm(goal - pos), 1e-6)
        self.agents.append(Pedestrian(
            id=self.next_agent_id, config=self.config,
            phone_user=self.rng.random() < self.args.p_phone,
            pos=pos, vel=heading * v0 * 0.5, v0=v0, goal=goal, spawn_time=self.sim_time,
        ))
        self.next_agent_id += 1

    # -------------------- 力の計算・状態更新 --------------------
    def _update_agents(self):
        for agent in self.agents:
            agent.history.append((self.sim_time, agent.pos.copy()))

        current_frame_overlaps = set(); current_frame_near = set(); current_frame_obstacle = set()
        forces = {agent.id: np.zeros(2) for agent in self.agents}

        for i, agent in enumerate(self.agents):
            # --- 注意ラプス判定（歩きスマホのみ発生。画面に視線が固定され
            #     周囲を一切見ていない瞬間。出典[6]） ---
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

            # --- 障害物（柱・キオスク）との相互作用 ---
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

            to_goal = agent.goal - agent.pos
            dist_goal = max(np.linalg.norm(to_goal), 1e-6)
            e0 = to_goal / dist_goal
            v_desired_vec = e0 * agent.v_desired_magnitude * speed_dampening_factor
            f_drive = (v_desired_vec - agent.vel) / self.config.RELAXATION_TIME_TAU_S
            forces[agent.id] = f_drive + f_avoid + f_obs + f_fluct

        for agent in self.agents:
            agent.vel += forces[agent.id] * self.dt
            speed = np.linalg.norm(agent.vel)
            if speed > self.config.MAX_SPEED_M_S:
                agent.vel *= self.config.MAX_SPEED_M_S / speed
            agent.pos += agent.vel * self.dt

        self.ongoing_collisions = current_frame_overlaps
        self.ongoing_near = current_frame_near
        self.ongoing_obstacle = current_frame_obstacle

        margin = 6.0
        despawn_candidates = [a for a in self.agents if (self.sim_time - a.spawn_time > 1.0) and (
            a.pos[0] < -margin or a.pos[0] > self.config.WIDTH_M + margin or
            a.pos[1] < -margin or a.pos[1] > self.config.HEIGHT_M + margin)]
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

    # -------------------- 描画 --------------------
    def _draw(self):
        c = self.config
        self.screen.fill(c.FLOOR_COLOR)
        for x in range(0, c.WIDTH_PX, 40):
            pygame.draw.line(self.screen, c.FLOOR_LINE_COLOR, (x, 0), (x, c.HEIGHT_PX), 1)
        for y in range(0, c.HEIGHT_PX, 40):
            pygame.draw.line(self.screen, c.FLOOR_LINE_COLOR, (0, y), (c.WIDTH_PX, y), 1)
        for gx in c.GATE_XS_M:
            px = int(gx * c.M_TO_PX)
            pygame.draw.rect(self.screen, c.GATE_COLOR, pygame.Rect(px - 9, 0, 18, 26))
        for (ox, oy, orad) in c.OBSTACLES:
            pygame.draw.circle(self.screen, c.OBSTACLE_COLOR, (int(ox * c.M_TO_PX), int(oy * c.M_TO_PX)), int(orad * c.M_TO_PX))

        for agent in self.agents:
            agent.draw(self.screen, self.sim_time < agent.lapse_until)

        self.collision_markers = [m for m in self.collision_markers if self.sim_time - m[2] < 1.2]
        marker_color = {"collision": None, "obstacle": c.OBSTACLE_MARKER_COLOR, "stumble": c.STUMBLE_MARKER_COLOR}
        for kind, pos, timestamp, danger in self.collision_markers:
            age = self.sim_time - timestamp
            alpha = max(0, 255 * (1 - age / 1.2))
            color = (c.COLLISION_MARKER_COLOR if danger else (255, 210, 63)) if kind == "collision" else marker_color[kind]
            radius = 15 + int(age * 20)
            surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*color, int(alpha)), (radius + 2, radius + 2), radius, width=3)
            px_pos = (int(pos[0] * c.M_TO_PX) - radius - 2, int(pos[1] * c.M_TO_PX) - radius - 2)
            self.screen.blit(surf, px_pos)

        info_lines = [
            f"Time: {self.sim_time:.1f}s",
            f"通過: {len(self.completed_agents)}  衝突: {self.total_collisions}  柱接触: {self.total_obstacle_collisions}",
            f"よろめき: {self.total_stumbles}  ニアミス: {self.total_near_misses}",
        ]
        for k, line in enumerate(info_lines):
            text = self.font.render(line, True, (232, 235, 239))
            self.screen.blit(text, (10, 8 + k * 22))
        if self.is_paused:
            paused_text = self.big_font.render("PAUSED", True, (255, 255, 0))
            self.screen.blit(paused_text, paused_text.get_rect(center=self.screen.get_rect().center))
        pygame.display.flip()

    def _write_live_data(self):
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
            with open("live_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f)
        except IOError as e:
            print(f"Error writing live data: {e}")

    def run(self):
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    self.is_running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
            if not self.is_paused:
                for _ in range(self.speed_multiplier):
                    self._spawn_pedestrian(); self._update_agents(); self.sim_time += self.dt
            self._draw()
            if int(self.sim_time * 10) % 2 == 0:
                self._write_live_data()
            self.clock.tick(60)
        pygame.quit(); sys.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_phone', type=float, default=0.3)
    parser.add_argument('--spawn_rate', type=float, default=1.5, help='駅コンコースを想定し既定値を引き上げ')
    parser.add_argument('--speed', type=int, default=1, help='Simulation speed multiplier.')
    parser.add_argument('--seed', type=int, default=None, help='再現したい場合のみ指定。未指定なら毎回変化する。')
    sim = Simulation(parser.parse_args())
    sim.run()


if __name__ == '__main__':
    main()
