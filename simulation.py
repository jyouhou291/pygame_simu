# =============================================================================
# 歩きスマホ行動シミュレーション (v16.2 - Simulation Engine)
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

# --- 定数・設定クラス (変更なし) ---
@dataclass
class Config:
    WIDTH_M: float = 30.0; SIDEWALK_HEIGHT_M: float = 2.5; ROAD_HEIGHT_M: float = 4.0
    AGENT_RADIUS_M: float = 0.3; V0_MEAN: float = 1.3; V0_STD: float = 0.2; MAX_SPEED_M_S: float = 2.0
    REACTION_TIME_DEFAULT_S: float = 0.2; REACTION_TIME_PHONE_S: float = 0.556
    FOV_DEFAULT_DEG: float = 120.0; FOV_PHONE_DEG: float = 90.0
    AVOID_BETA_PHONE: float = 0.7; SPEED_ALPHA_PHONE: float = 0.85
    RELAXATION_TIME_TAU_S: float = 0.5; AGENT_REPULSION_A: float = 2.1
    AGENT_REPULSION_B_PHONE: float = 0.3; AGENT_REPULSION_B_NORMAL: float = 2.0
    WALL_REPULSION_A: float = 3.0; WALL_REPULSION_B: float = 0.1
    NORMAL_EVASION_BOOST: float = 1.8; SHUFFLE_EVASION_BOOST: float = 0.5; SHOULDER_PASS_BOOST: float = 0.9
    FORWARD_EVASION_BOOST: float = 1.5
    M_TO_PX: int = 25; FOV_VIS_RADIUS_M: float = 1.0
    SIDEWALK_COLOR: tuple = (200, 200, 200); ROAD_COLOR: tuple = (105, 105, 105); LINE_COLOR: tuple = (255, 255, 100)
    AGENT_COLOR: tuple = (0, 100, 200); AGENT_PHONE_COLOR: tuple = (220, 50, 50); COLLISION_MARKER_COLOR: tuple = (255, 0, 0)
    TOTAL_HEIGHT_M: float = field(init=False); WIDTH_PX: int = field(init=False); HEIGHT_PX: int = field(init=False)
    def __post_init__(self):
        self.TOTAL_HEIGHT_M = 2 * self.SIDEWALK_HEIGHT_M + self.ROAD_HEIGHT_M
        self.WIDTH_PX = int(self.WIDTH_M * self.M_TO_PX)
        self.HEIGHT_PX = int(self.TOTAL_HEIGHT_M * self.M_TO_PX)

# --- エージェントクラス (変更なし) ---
@dataclass
class Pedestrian:
    id: int; config: Config; phone_user: bool; pos: np.ndarray; vel: np.ndarray; v0: float; direction: int
    radius: float = field(init=False); reaction_time: float = field(init=False); fov_rad: float = field(init=False); avoid_beta: float = field(init=False); v_desired_magnitude: float = field(init=False)
    history: deque = field(default_factory=lambda: deque(maxlen=60)); spawn_time: float = 0.0; exit_time: float = -1.0; collisions_count: int = 0
    def __post_init__(self):
        self.radius = self.config.AGENT_RADIUS_M
        if self.phone_user: self.reaction_time, self.fov_rad, self.avoid_beta, self.v_desired_magnitude = self.config.REACTION_TIME_PHONE_S, np.deg2rad(self.config.FOV_PHONE_DEG), self.config.AVOID_BETA_PHONE, self.v0 * self.config.SPEED_ALPHA_PHONE
        else: self.reaction_time, self.fov_rad, self.avoid_beta, self.v_desired_magnitude = self.config.REACTION_TIME_DEFAULT_S, np.deg2rad(self.config.FOV_DEFAULT_DEG), 1.0, self.v0
    def get_delayed_state(self, t: float) -> np.ndarray:
        tt=t-self.reaction_time; 
        if not self.history or tt<=self.history[0][0]: return self.history[0][1] if self.history else self.pos
        for i in range(len(self.history)-1): 
            if self.history[i][0]<=tt<self.history[i+1][0]: return self.history[i][1]
        return self.history[-1][1]
    def draw(self, s: pygame.Surface):
        px_pos=(int(self.pos[0]*self.config.M_TO_PX), int(self.pos[1]*self.config.M_TO_PX)); color=self.config.AGENT_PHONE_COLOR if self.phone_user else self.config.AGENT_COLOR
        pygame.draw.circle(s, color, px_pos, int(self.config.AGENT_RADIUS_M * self.config.M_TO_PX))
        if np.linalg.norm(self.vel)>0.1:
            a=math.atan2(-self.vel[1],self.vel[0]); r=int(self.config.FOV_VIS_RADIUS_M*self.config.M_TO_PX); rect=pygame.Rect(px_pos[0]-r,px_pos[1]-r,2*r,2*r); sa,ea=a-self.fov_rad/2,a+self.fov_rad/2; fov_c=tuple(min(255,c+80) for c in color)
            try: pygame.draw.arc(s,fov_c,rect,sa,ea,1); pygame.draw.line(s,fov_c,px_pos,(px_pos[0]+r*math.cos(sa),px_pos[1]-r*math.sin(sa)),1); pygame.draw.line(s,fov_c,px_pos,(px_pos[0]+r*math.cos(ea),px_pos[1]-r*math.sin(ea)),1)
            except TypeError: pass

# --- シミュレーション本体クラス ---
class Simulation:
    def __init__(self, args):
        self.args = args; self.config = Config(); np.random.seed(42)
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.WIDTH_PX, self.config.HEIGHT_PX))
        pygame.display.set_caption("歩きスマホ行動シミュレーション")
        self.font = pygame.font.Font(None, 32); self.big_font = pygame.font.Font(None, 72)
        self.clock = pygame.time.Clock(); self.dt = 1.0 / 60.0
        self.is_running = True; self.sim_time = 0.0; self.is_paused = False
        self.speed_multiplier = args.speed
        self.agents = []; self.completed_agents = []; self.next_agent_id = 0
        self.total_collisions = 0
        self.phone_users_completed = 0; self.normal_users_completed = 0
        self.phone_user_total_collisions = 0; self.normal_user_total_collisions = 0
        self.collision_markers = []; self.ongoing_collisions = set()
        self._define_walkable_areas()

    def _define_walkable_areas(self):
        self.sidewalk_top_y_range_m = (0, self.config.SIDEWALK_HEIGHT_M)
        self.sidewalk_bottom_y_range_m = (self.config.SIDEWALK_HEIGHT_M + self.config.ROAD_HEIGHT_M, self.config.TOTAL_HEIGHT_M)

    def _spawn_pedestrian(self):
        if np.random.rand() < self.args.spawn_rate * self.dt:
            direction = 1 if np.random.rand() < 0.5 else -1
            y_range = self.sidewalk_top_y_range_m if np.random.rand() < 0.5 else self.sidewalk_bottom_y_range_m
            # 修正点1: スポーン位置を少し広げる
            if direction == 1:
                pos_x = np.random.uniform(self.config.AGENT_RADIUS_M, self.config.AGENT_RADIUS_M * 3)
            else:
                pos_x = np.random.uniform(self.config.WIDTH_M - self.config.AGENT_RADIUS_M * 3, self.config.WIDTH_M - self.config.AGENT_RADIUS_M)
            pos = np.array([pos_x, np.random.uniform(y_range[0] + self.config.AGENT_RADIUS_M, y_range[1] - self.config.AGENT_RADIUS_M)])
            if any(np.linalg.norm(pos - a.pos) < 2 * self.config.AGENT_RADIUS_M for a in self.agents): return
            v0 = np.random.normal(self.config.V0_MEAN, self.config.V0_STD)
            self.agents.append(Pedestrian(id=self.next_agent_id, config=self.config, phone_user=np.random.rand()<self.args.p_phone, pos=pos, vel=np.array([v0*direction,0]), v0=v0, direction=direction, spawn_time=self.sim_time)); self.next_agent_id+=1

    def _update_agents(self):
        for agent in self.agents: agent.history.append((self.sim_time, agent.pos.copy()))
        current_frame_overlaps = set()
        forces = {agent.id: np.zeros(2) for agent in self.agents}

        for i, agent in enumerate(self.agents):
            f_avoid = np.zeros(2)
            speed_dampening_factor = 1.0
            for j, other in enumerate(self.agents):
                if i == j: continue
                vec_to_other = other.get_delayed_state(self.sim_time) - agent.get_delayed_state(self.sim_time); dist = np.linalg.norm(vec_to_other)
                
                if dist < agent.radius + other.radius:
                    pair = tuple(sorted((agent.id, other.id)))
                    current_frame_overlaps.add(pair)
                    if pair not in self.ongoing_collisions:
                        agent.collisions_count += 1; other.collisions_count += 1; self.total_collisions += 1
                        self.collision_markers.append((agent.pos + vec_to_other / 2, self.sim_time))
                
                is_in_fov = True
                if np.linalg.norm(agent.vel) > 0.1 and dist > 0:
                    if abs(np.arccos(np.clip(np.dot(agent.vel/np.linalg.norm(agent.vel), vec_to_other/dist), -1.0, 1.0))) > agent.fov_rad/2:
                        is_in_fov = False
                
                if is_in_fov:
                    if dist < self.config.AGENT_RADIUS_M * 4: speed_dampening_factor = max(0.6, speed_dampening_factor - 0.1)
                    repulsion_b = self.config.AGENT_REPULSION_B_NORMAL if not agent.phone_user else self.config.AGENT_REPULSION_B_PHONE
                    force_mag = self.config.AGENT_REPULSION_A * np.exp((agent.radius + other.radius - dist) / repulsion_b)
                    evasion_boost = 1.0
                    if not agent.phone_user: # Agent is blue
                        use_default_logic = True
                        # Special case for blue-on-blue interaction
                        if not other.phone_user:
                            is_red_nearby = any(
                                red.phone_user and np.linalg.norm(agent.pos - red.pos) < 7.0
                                for red in self.agents
                            )
                            if not is_red_nearby:
                                evasion_boost = self.config.SHOULDER_PASS_BOOST # Shoulder-to-shoulder
                                use_default_logic = False

                        if use_default_logic:
                            relative_vel = other.vel - agent.vel
                            if np.dot(vec_to_other, relative_vel) < 0:
                                if np.random.rand() < (1/3):
                                    evasion_boost = self.config.SHUFFLE_EVASION_BOOST
                                else:
                                    evasion_boost = self.config.NORMAL_EVASION_BOOST
                            else:
                                evasion_boost = self.config.NORMAL_EVASION_BOOST
                    
                    # --- 前方注意ブースト (青エージェントのみ) ---
                    if not agent.phone_user and np.linalg.norm(agent.vel) > 0.1 and dist > 0:
                        # 自分の進行方向と相手へのベクトルのなす角度のコサインを計算
                        cos_angle = np.dot(agent.vel, vec_to_other) / (np.linalg.norm(agent.vel) * dist)
                        if cos_angle > 0.9: # 進行方向の約25度以内に相手がいる場合
                            evasion_boost *= self.config.FORWARD_EVASION_BOOST
                            
                    force_direction = -vec_to_other / dist if dist > 0 else np.zeros(2)
                    f_avoid += agent.avoid_beta * evasion_boost * force_mag * force_direction
            
            v_desired_vec = np.array([agent.v_desired_magnitude * agent.direction * speed_dampening_factor, 0])
            f_drive = (v_desired_vec - agent.vel) / self.config.RELAXATION_TIME_TAU_S
            y_range = self.sidewalk_top_y_range_m if agent.pos[1] < self.config.TOTAL_HEIGHT_M / 2 else self.sidewalk_bottom_y_range_m
            f_wall = np.zeros(2)
            f_wall += np.array([0, self.config.WALL_REPULSION_A * np.exp((agent.radius - (agent.pos[1] - y_range[0])) / self.config.WALL_REPULSION_B)])
            f_wall += np.array([0, -self.config.WALL_REPULSION_A * np.exp((agent.radius - (y_range[1] - agent.pos[1])) / self.config.WALL_REPULSION_B)])
            forces[agent.id] = f_drive + f_avoid + f_wall

        for agent in self.agents:
            agent.vel += forces[agent.id] * self.dt
            speed = np.linalg.norm(agent.vel)
            if speed > self.config.MAX_SPEED_M_S: agent.vel *= self.config.MAX_SPEED_M_S / speed
            agent.pos += agent.vel * self.dt
            # 修正点2: X座標も画面内にクリップする
            agent.pos[0] = np.clip(agent.pos[0], 0, self.config.WIDTH_M)
            y_range = self.sidewalk_top_y_range_m if agent.pos[1] < self.config.TOTAL_HEIGHT_M / 2 else self.sidewalk_bottom_y_range_m
            agent.pos[1] = np.clip(agent.pos[1], y_range[0] + agent.radius, y_range[1] - agent.radius)

        self.ongoing_collisions = current_frame_overlaps
        despawn_candidates = [a for a in self.agents if (self.sim_time - a.spawn_time > 2.0) and ((a.direction==1 and a.pos[0]>=self.config.WIDTH_M)or(a.direction==-1 and a.pos[0]<=0))]
        for agent in despawn_candidates:
            agent.exit_time=self.sim_time; self.completed_agents.append(agent)
            if agent.phone_user: self.phone_users_completed += 1; self.phone_user_total_collisions += agent.collisions_count
            else: self.normal_users_completed += 1; self.normal_user_total_collisions += agent.collisions_count
        self.agents=[a for a in self.agents if a.exit_time<0]

    def _draw(self):
        self.screen.fill(self.config.ROAD_COLOR)
        pygame.draw.rect(self.screen,self.config.SIDEWALK_COLOR,pygame.Rect(0,0,self.config.WIDTH_PX,self.config.SIDEWALK_HEIGHT_M*self.config.M_TO_PX))
        pygame.draw.rect(self.screen,self.config.SIDEWALK_COLOR,pygame.Rect(0,(self.config.SIDEWALK_HEIGHT_M+self.config.ROAD_HEIGHT_M)*self.config.M_TO_PX,self.config.WIDTH_PX,self.config.SIDEWALK_HEIGHT_M*self.config.M_TO_PX))
        for x in range(0,self.config.WIDTH_PX,40): pygame.draw.line(self.screen,self.config.LINE_COLOR,(x,self.config.TOTAL_HEIGHT_M/2*self.config.M_TO_PX),(x+20,self.config.TOTAL_HEIGHT_M/2*self.config.M_TO_PX),3)
        for agent in self.agents: agent.draw(self.screen)
        self.collision_markers = [m for m in self.collision_markers if self.sim_time - m[1] < 1.5]
        for pos, timestamp in self.collision_markers:
            alpha = max(0, 255 * (1 - (self.sim_time - timestamp) / 1.5))
            marker_surface = pygame.Surface((30, 30), pygame.SRCALPHA)
            pygame.draw.circle(marker_surface, (*self.config.COLLISION_MARKER_COLOR, int(alpha)), (15, 15), 15)
            px_pos = (int(pos[0] * self.config.M_TO_PX) - 15, int(pos[1] * self.config.M_TO_PX) - 15)
            self.screen.blit(marker_surface, px_pos)
        time_text = self.font.render(f"Time: {self.sim_time:.1f}s", True, (255, 255, 255)); self.screen.blit(time_text, (10, 10))
        if self.is_paused:
            paused_text = self.big_font.render("PAUSED", True, (255, 255, 0)); text_rect = paused_text.get_rect(center=self.screen.get_rect().center); self.screen.blit(paused_text, text_rect)
        pygame.display.flip()

    def _write_live_data(self):
        current_avg_speed = np.mean([np.linalg.norm(a.vel) for a in self.agents]) if self.agents else 0
        rate_phone = self.phone_user_total_collisions / self.phone_users_completed if self.phone_users_completed > 0 else 0
        rate_normal = self.normal_user_total_collisions / self.normal_users_completed if self.normal_users_completed > 0 else 0
        data = {"time": self.sim_time, "agent_count": len(self.agents), "completed_count": len(self.completed_agents), "total_collisions": self.total_collisions, "avg_speed": current_avg_speed, "collision_rate_phone": rate_phone, "collision_rate_normal": rate_normal}
        try:
            with open("live_data.json", "w") as f: json.dump(data, f)
        except IOError as e: print(f"Error writing live data: {e}")

    def run(self):
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q): self.is_running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: self.is_paused = not self.is_paused
            if not self.is_paused:
                for _ in range(self.speed_multiplier):
                    self._spawn_pedestrian(); self._update_agents(); self.sim_time += self.dt
            self._draw()
            if int(self.sim_time * 10) % 2 == 0: self._write_live_data()
            self.clock.tick(60)
        pygame.quit(); sys.exit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_phone', type=float, default=0.3); parser.add_argument('--spawn_rate', type=float, default=1.0)
    parser.add_argument('--speed', type=int, default=1, help='Simulation speed multiplier.')
    sim = Simulation(parser.parse_args())
    sim.run()

if __name__ == '__main__':
    main()
