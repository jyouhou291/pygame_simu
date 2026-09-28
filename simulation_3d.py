# =============================================================================
# 歩きスマホ行動シミュレーション ― 3D版 (Ursina Engine)
#
# 物理演算・イベント検出は sim_core.py（2D版と完全共通）をそのまま使う。
# このファイルは「同じシミュレーションを3Dで見せる」レンダラーに専念する。
#
# 操作方法:
#   ← →キー        : 視点を回転（真上から見た円周方向）
#   ↑ ↓キー        : 見下ろす角度を調整
#   + / - キー      : ズームイン・アウト
#   Space キー      : 一時停止
#   Q キー          : 終了
#
# 必要ライブラリ: pip install ursina
# =============================================================================
import argparse
import math
import os

from ursina import (
    Ursina, Entity, Text, Mesh, Vec3, color,
    destroy, time as ursina_time, held_keys, application, camera, window,
)
from ursina.models.procedural.cylinder import Cylinder

from sim_core import SimulationCore


def make_fov_mesh(angle_deg: float, radius: float, segments: int = 10):
    """進行方向を中心とした扇形（視野）を表す三角形ファンのメッシュをXZ平面に作る。
    ローカルの前方向は +z とする。"""
    half = math.radians(angle_deg) / 2.0
    verts = [Vec3(0, 0, 0)]
    for i in range(segments + 1):
        a = -half + (2 * half) * (i / segments)
        verts.append(Vec3(math.sin(a) * radius, 0, math.cos(a) * radius))
    tris = []
    for i in range(1, segments + 1):
        tris += [0, i, i + 1]
    return Mesh(vertices=verts, triangles=tris, mode='triangle')


class AgentView:
    """1人の歩行者に対応する3Dエンティティ一式（胴体・頭・視野扇形）。"""

    def __init__(self, agent, config):
        is_phone = agent.phone_user
        c = color.rgb32(220, 50, 50) if is_phone else color.rgb32(40, 110, 210)
        self.root = Entity(position=(agent.pos[0], 0, agent.pos[1]))
        self.body = Entity(parent=self.root,
                            model=Cylinder(resolution=8, radius=config.AGENT_RADIUS_M * 0.8, height=1.1),
                            color=c, position=(0, 0.0, 0))
        self.head = Entity(parent=self.root, model='sphere', color=c,
                            scale=config.AGENT_RADIUS_M * 1.3, position=(0, 1.25, 0))
        if is_phone:
            # スマホを持つ手元を表す小さな白い板
            self.phone = Entity(parent=self.root, model='cube', color=color.white,
                                 scale=(0.12, 0.2, 0.03), position=(0.25, 0.9, 0.25),
                                 rotation=(20, -30, 0))
        fov_angle = 90.0 if is_phone else 120.0
        fov_radius = 1.6 if is_phone else 3.2
        fov_alpha = 130 if is_phone else 28
        self.fov = Entity(parent=self.root, model=make_fov_mesh(fov_angle, fov_radius),
                           color=color.rgba(c[0], c[1], c[2], fov_alpha / 255),
                           position=(0, 0.03, 0), double_sided=True)
        self.lapse_dot = Entity(parent=self.root, model='sphere', color=color.white,
                                 scale=0.12, position=(0, 1.7, 0), enabled=False)

    def update_from(self, agent, in_lapse: bool):
        self.root.position = Vec3(agent.pos[0], 0, agent.pos[1])
        speed = (agent.vel[0] ** 2 + agent.vel[1] ** 2) ** 0.5
        if speed > 0.1:
            yaw = math.degrees(math.atan2(agent.vel[0], agent.vel[1]))
            self.root.rotation_y = yaw
        self.fov.enabled = not in_lapse
        self.lapse_dot.enabled = in_lapse

    def destroy(self):
        destroy(self.root)  # 子エンティティも連鎖して破棄される


class MarkerView:
    """衝突・柱接触・よろめきの発生地点を示す、広がって消えるリング。"""
    COLORS = {
        "collision_danger": color.rgb32(255, 59, 48),
        "collision_safe": color.rgb32(255, 210, 63),
        "obstacle": color.rgb32(255, 176, 32),
        "stumble": color.rgb32(176, 125, 255),
    }

    def __init__(self, kind: str, x: float, z: float, danger: bool):
        key = "collision_danger" if (kind == "collision" and danger) else \
              "collision_safe" if kind == "collision" else kind
        c = self.COLORS.get(key, color.white)
        self.entity = Entity(model='circle', color=color.rgba(c[0], c[1], c[2], 220 / 255),
                              position=(x, 0.05, z), rotation_x=90, scale=0.6,
                              double_sided=True)
        self.age = 0.0

    def tick(self, dt: float) -> bool:
        """経過時間を進める。生存していれば True を返す。"""
        self.age += dt
        t = min(1.0, self.age / 1.0)
        self.entity.scale = 0.6 + t * 2.2
        c = self.entity.color
        self.entity.color = color.rgba(c[0], c[1], c[2], 220 / 255 * (1 - t))
        if self.age >= 1.0:
            destroy(self.entity)
            return False
        return True


def build_scene(core: SimulationCore):
    c = core.config
    W, H = c.WIDTH_M, c.TOTAL_HEIGHT_M

    ground_parent = Entity()
    # 車道
    Entity(parent=ground_parent, model='cube', color=color.rgb32(*c.ROAD_COLOR),
           position=(W / 2, -0.05, c.SIDEWALK_HEIGHT_M + c.ROAD_HEIGHT_M / 2),
           scale=(W, 0.1, c.ROAD_HEIGHT_M))
    # 歩道（上・下）
    Entity(parent=ground_parent, model='cube', color=color.rgb32(*c.SIDEWALK_COLOR),
           position=(W / 2, 0.0, c.SIDEWALK_HEIGHT_M / 2), scale=(W, 0.12, c.SIDEWALK_HEIGHT_M))
    Entity(parent=ground_parent, model='cube', color=color.rgb32(*c.SIDEWALK_COLOR),
           position=(W / 2, 0.0, H - c.SIDEWALK_HEIGHT_M / 2), scale=(W, 0.12, c.SIDEWALK_HEIGHT_M))
    # 中央の破線
    center_z = H / 2
    n_dashes = 15
    for i in range(n_dashes):
        x = (i + 0.5) * (W / n_dashes)
        Entity(parent=ground_parent, model='cube', color=color.rgb32(255, 255, 100),
               position=(x, 0.01, center_z), scale=(W / n_dashes * 0.5, 0.02, 0.25))
    # 街灯・標識ポール
    for (ox, oy, orad) in c.OBSTACLES:
        pole = Entity(parent=ground_parent,
                      model=Cylinder(resolution=8, radius=orad, height=1.8),
                      color=color.rgb32(90, 90, 90), position=(ox, 0.0, oy))
        Entity(parent=pole, model='sphere', color=color.yellow.tint(-.1),
               position=(0, 1.85, 0), scale=(orad * 4, orad * 1.2, orad * 4))
    return ground_parent


def ensure_japanese_font() -> str | None:
    """HUD表示用の日本語フォントを用意する。Ursinaはプロジェクト直下(または
    fontsフォルダ)にあるファイルしか参照できないため、Windows同梱の
    meiryo.ttcが無ければこの場でコピーする（ライセンス上リポジトリには
    含めず、.gitignoreで除外している）。"""
    import shutil
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meiryo.ttc")
    if not os.path.exists(local_path):
        win_font = r"C:\Windows\Fonts\meiryo.ttc"
        if os.path.exists(win_font):
            try:
                shutil.copy(win_font, local_path)
            except OSError:
                return None
        else:
            return None
    return "meiryo.ttc"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_phone', type=float, default=0.3)
    parser.add_argument('--spawn_rate', type=float, default=1.0)
    parser.add_argument('--speed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    core = SimulationCore(p_phone=args.p_phone, spawn_rate=args.spawn_rate, seed=args.seed)
    speed_multiplier = args.speed

    app = Ursina(title="歩きスマホ行動シミュレーション（3D）", borderless=False)
    window.color = color.rgb32(15, 18, 22)

    build_scene(core)

    W, H = core.config.WIDTH_M, core.config.TOTAL_HEIGHT_M
    cam_target = Vec3(W / 2, 0, H / 2)
    camera.position = (W / 2, 16, -8)
    camera.look_at(cam_target)

    views = {}  # agent.id -> AgentView
    markers = []
    state = {"paused": False}

    jp_font = ensure_japanese_font()
    hud_kwargs = {"font": jp_font} if jp_font else {}
    hud = Text(text="", position=(-0.86, 0.46), scale=1.1, background=True, **hud_kwargs)

    def refresh_hud():
        hud.text = (
            f"Time: {core.sim_time:.1f}s\n"
            f"通過: {len(core.completed_agents)}  衝突: {core.total_collisions}  柱接触: {core.total_obstacle_collisions}\n"
            f"よろめき: {core.total_stumbles}  ニアミス: {core.total_near_misses}\n"
            f"[←→:回転 ↑↓:見下ろす角度 +/-:ズーム Space:一時停止 Q:終了]"
        )

    cam_state = {"yaw": -18.0, "pitch": 50.0, "dist": 24.0}

    def apply_camera():
        yaw = math.radians(cam_state["yaw"])
        pitch = math.radians(cam_state["pitch"])
        d = cam_state["dist"]
        offset = Vec3(math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch)) * d
        camera.position = cam_target + offset
        camera.look_at(cam_target)

    def update():
        if held_keys['q']:
            application.quit()

        # 簡易オービットカメラ（矢印キーで回転・上下、+/-でズーム）
        cam_changed = False
        if held_keys['left arrow']:
            cam_state["yaw"] -= 60 * ursina_time.dt; cam_changed = True
        if held_keys['right arrow']:
            cam_state["yaw"] += 60 * ursina_time.dt; cam_changed = True
        if held_keys['up arrow']:
            cam_state["pitch"] = min(85, cam_state["pitch"] + 40 * ursina_time.dt); cam_changed = True
        if held_keys['down arrow']:
            cam_state["pitch"] = max(5, cam_state["pitch"] - 40 * ursina_time.dt); cam_changed = True
        if held_keys['+'] or held_keys['=']:
            cam_state["dist"] = max(6, cam_state["dist"] - 15 * ursina_time.dt); cam_changed = True
        if held_keys['-']:
            cam_state["dist"] = min(45, cam_state["dist"] + 15 * ursina_time.dt); cam_changed = True
        if cam_changed:
            apply_camera()

        if not state["paused"]:
            for _ in range(speed_multiplier):
                core.step()

            seen_ids = set()
            for agent in core.agents:
                seen_ids.add(agent.id)
                if agent.id not in views:
                    views[agent.id] = AgentView(agent, core.config)
                views[agent.id].update_from(agent, core.sim_time < agent.lapse_until)
            for aid in list(views.keys()):
                if aid not in seen_ids:
                    views[aid].destroy()
                    del views[aid]

            for kind, pos, ts, danger in core.collision_markers:
                markers.append(MarkerView(kind, float(pos[0]), float(pos[1]), bool(danger)))
            core.collision_markers = []  # 3D側は都度消費するので、2Dのような年齢フィルタは不要

            for m in markers[:]:
                if not m.tick(ursina_time.dt):
                    markers.remove(m)

            if int(core.sim_time * 10) % 2 == 0:
                core.write_live_data()
            refresh_hud()

    def input(key):
        if key == 'space':
            state["paused"] = not state["paused"]

    apply_camera()
    controller = Entity()
    controller.update = update
    controller.input = input

    app.run()


if __name__ == '__main__':
    main()
