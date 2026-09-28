# =============================================================================
# 歩きスマホ行動シミュレーション (v19.0 - 2D Pygame レンダラー)
#
# v19.0: 物理演算ロジックを sim_core.py（Pygame非依存）に切り出した。
# 3D版 (simulation_3d.py, Ursina) と全く同じ物理エンジンを使っており、
# このファイルは「歩道を2Dで見た目に描画する」役割だけを持つ。
#
# v17.0 → v18.0 → v19.0 の変更履歴・パラメータの根拠は sim_core.py と
# SPECIFICATION.md を参照。
# =============================================================================
import pygame
import argparse
import math
import sys
import numpy as np

from sim_core import SimulationCore, Config


def draw_agent(surface, config: Config, agent, in_lapse: bool):
    px_pos = (int(agent.pos[0] * config.M_TO_PX), int(agent.pos[1] * config.M_TO_PX))
    color = config.AGENT_PHONE_COLOR if agent.phone_user else config.AGENT_COLOR
    pygame.draw.circle(surface, color, px_pos, int(config.AGENT_RADIUS_M * config.M_TO_PX))
    if np.linalg.norm(agent.vel) > 0.1 and not in_lapse:
        a = math.atan2(-agent.vel[1], agent.vel[0])
        r = int(config.FOV_VIS_RADIUS_M * config.M_TO_PX * (0.7 if agent.phone_user else 1.3))
        rect = pygame.Rect(px_pos[0] - r, px_pos[1] - r, 2 * r, 2 * r)
        sa, ea = a - agent.fov_rad / 2, a + agent.fov_rad / 2
        fov_c = tuple(min(255, c + 60) for c in color)
        try:
            pygame.draw.arc(surface, fov_c, rect, sa, ea, 1)
            pygame.draw.line(surface, fov_c, px_pos, (px_pos[0] + r * math.cos(sa), px_pos[1] - r * math.sin(sa)), 1)
            pygame.draw.line(surface, fov_c, px_pos, (px_pos[0] + r * math.cos(ea), px_pos[1] - r * math.sin(ea)), 1)
        except TypeError:
            pass
    elif in_lapse:
        pygame.draw.circle(surface, (255, 255, 255), (px_pos[0], px_pos[1] - 14), 2)


class PygameRenderer:
    """SimulationCore の物理状態を Pygame で描画するだけのラッパー。"""

    def __init__(self, args):
        self.core = SimulationCore(p_phone=args.p_phone, spawn_rate=args.spawn_rate, seed=args.seed)
        self.speed_multiplier = args.speed
        c = self.core.config
        pygame.init()
        self.screen = pygame.display.set_mode((c.WIDTH_PX, c.HEIGHT_PX))
        pygame.display.set_caption("歩きスマホ行動シミュレーション")
        self.font = pygame.font.Font(None, 26); self.big_font = pygame.font.Font(None, 72)
        self.clock = pygame.time.Clock()
        self.is_running = True; self.is_paused = False

    def _draw(self):
        core = self.core; c = core.config
        self.screen.fill(c.ROAD_COLOR)
        pygame.draw.rect(self.screen, c.SIDEWALK_COLOR, pygame.Rect(0, 0, c.WIDTH_PX, int(c.SIDEWALK_HEIGHT_M * c.M_TO_PX)))
        pygame.draw.rect(self.screen, c.SIDEWALK_COLOR, pygame.Rect(
            0, int((c.SIDEWALK_HEIGHT_M + c.ROAD_HEIGHT_M) * c.M_TO_PX), c.WIDTH_PX, int(c.SIDEWALK_HEIGHT_M * c.M_TO_PX)))
        center_y = int(c.TOTAL_HEIGHT_M / 2 * c.M_TO_PX)
        for x in range(0, c.WIDTH_PX, 40):
            pygame.draw.line(self.screen, c.LINE_COLOR, (x, center_y), (x + 20, center_y), 3)
        for (ox, oy, orad) in c.OBSTACLES:
            pygame.draw.circle(self.screen, c.OBSTACLE_COLOR, (int(ox * c.M_TO_PX), int(oy * c.M_TO_PX)), max(3, int(orad * c.M_TO_PX)))

        for agent in core.agents:
            draw_agent(self.screen, c, agent, core.sim_time < agent.lapse_until)

        core.collision_markers = [m for m in core.collision_markers if core.sim_time - m[2] < 1.2]
        marker_color = {"collision": None, "obstacle": c.OBSTACLE_MARKER_COLOR, "stumble": c.STUMBLE_MARKER_COLOR}
        for kind, pos, timestamp, danger in core.collision_markers:
            age = core.sim_time - timestamp
            alpha = max(0, 255 * (1 - age / 1.2))
            color = (c.COLLISION_MARKER_COLOR if danger else (255, 210, 63)) if kind == "collision" else marker_color[kind]
            radius = 15 + int(age * 20)
            surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*color, int(alpha)), (radius + 2, radius + 2), radius, width=3)
            px_pos = (int(pos[0] * c.M_TO_PX) - radius - 2, int(pos[1] * c.M_TO_PX) - radius - 2)
            self.screen.blit(surf, px_pos)

        info_lines = [
            f"Time: {core.sim_time:.1f}s",
            f"通過: {len(core.completed_agents)}  衝突: {core.total_collisions}  柱接触: {core.total_obstacle_collisions}",
            f"よろめき: {core.total_stumbles}  ニアミス: {core.total_near_misses}",
        ]
        for k, line in enumerate(info_lines):
            text = self.font.render(line, True, (232, 235, 239))
            self.screen.blit(text, (10, 8 + k * 22))
        if self.is_paused:
            paused_text = self.big_font.render("PAUSED", True, (255, 255, 0))
            self.screen.blit(paused_text, paused_text.get_rect(center=self.screen.get_rect().center))
        pygame.display.flip()

    def run(self):
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    self.is_running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
            if not self.is_paused:
                for _ in range(self.speed_multiplier):
                    self.core.step()
            self._draw()
            if int(self.core.sim_time * 10) % 2 == 0:
                self.core.write_live_data()
            self.clock.tick(60)
        pygame.quit(); sys.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_phone', type=float, default=0.3)
    parser.add_argument('--spawn_rate', type=float, default=1.0)
    parser.add_argument('--speed', type=int, default=1, help='Simulation speed multiplier.')
    parser.add_argument('--seed', type=int, default=None, help='再現したい場合のみ指定。未指定なら毎回変化する。')
    renderer = PygameRenderer(parser.parse_args())
    renderer.run()


if __name__ == '__main__':
    main()
