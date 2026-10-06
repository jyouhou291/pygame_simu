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
from __future__ import annotations
import argparse
import math
import os

from ursina import (
    Ursina, Entity, Text, Mesh, Vec3, color,
    destroy, time as ursina_time, held_keys, application, camera, window,
    DirectionalLight, AmbientLight, mouse,
)
from ursina.models.procedural.cylinder import Cylinder

from sim_core import SimulationCore


# トレースログ解析の結果、速度が低い瞬間（渋滞・電柱回避でほぼ停止する時）ほど
# 社会的力モデルの微小なノイズで進行方向(atan2)が激しく暴れることが分かった
# （速度0.6m/s未満のサンプルだけで、300度/秒を超える異常な急旋回の99%が発生）。
# 速度がほぼゼロのときは直前の向きを維持（凍結）し、見た目の暴れを防ぐ。
YAW_FREEZE_MIN_SPEED = 0.5


def lerp_angle_deg(a: float, b: float, t: float) -> float:
    """最短経路で角度(度)を補間する（180度またぎの逆回りを防ぐ）。"""
    diff = (b - a + 180) % 360 - 180
    return a + diff * t


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
        self.is_phone = is_phone
        self.agent_id = agent.id
        c = color.rgb(220, 50, 50) if is_phone else color.rgb(40, 110, 210)
        self.root = Entity(position=(agent.pos[0], 0, agent.pos[1]))
        # クリックでこの人の一人称視点に入れるようにするための当たり判定
        # （体・頭そのものだと的が小さすぎてクリックしづらいので、少し
        # 大きめの非表示ボックスをかぶせる）
        self.click_target = Entity(parent=self.root, model='cube', collider='box', visible=False,
                                    scale=(0.7, 1.8, 0.7), position=(0, 0.9, 0))
        self.click_target.agent_id = agent.id
        # 胴体・頭は別Entityにまとめ、歩行の上下ゆれ(bob)をここに掛ける
        self.torso = Entity(parent=self.root, position=(0, 0, 0))
        self.body = Entity(parent=self.torso,
                            model=Cylinder(resolution=8, radius=config.AGENT_RADIUS_M * 0.8, height=1.1),
                            color=c, position=(0, 0.0, 0),
                            rotation_x=12 if is_phone else 0)  # スマホ利用者は少し前かがみ
        self.head = Entity(parent=self.torso, model='sphere', color=c,
                            scale=config.AGENT_RADIUS_M * 1.3, position=(0, 1.25, 0),
                            rotation_x=28 if is_phone else 0)  # スマホ利用者は俯く
        if is_phone:
            # スマホを持つ手元（前かがみのポーズに合わせてやや大きめ・前方に）
            self.phone = Entity(parent=self.torso, model='cube', color=color.white,
                                 scale=(0.18, 0.28, 0.03), position=(0.22, 1.0, 0.32),
                                 rotation=(35, -25, 0))
        self.phase = agent.id * 1.7  # 個体差をつけて全員同期しないようにする
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
        if speed > YAW_FREEZE_MIN_SPEED:
            yaw = math.degrees(math.atan2(agent.vel[0], agent.vel[1]))
            self.root.rotation_y = yaw
        # 歩行の上下ゆれ（速度に応じて周期が変わる、簡易的な"歩いている感"の演出）
        self.phase += speed * ursina_time.dt * 5.0
        self.torso.y = abs(math.sin(self.phase)) * 0.05
        self.fov.enabled = not in_lapse
        self.lapse_dot.enabled = in_lapse

    def destroy(self):
        destroy(self.root)  # 子エンティティも連鎖して破棄される


class MarkerView:
    """衝突・柱接触・よろめきの発生地点を示す、広がって消えるリング。"""
    COLORS = {
        "collision_danger": color.rgb(255, 59, 48),
        "collision_safe": color.rgb(255, 210, 63),
        "obstacle": color.rgb(255, 176, 32),
        "stumble": color.rgb(176, 125, 255),
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
    Entity(parent=ground_parent, model='cube', color=color.rgb(*c.ROAD_COLOR),
           position=(W / 2, -0.05, c.SIDEWALK_HEIGHT_M + c.ROAD_HEIGHT_M / 2),
           scale=(W, 0.1, c.ROAD_HEIGHT_M))
    # 歩道（上・下）
    Entity(parent=ground_parent, model='cube', color=color.rgb(*c.SIDEWALK_COLOR),
           position=(W / 2, 0.0, c.SIDEWALK_HEIGHT_M / 2), scale=(W, 0.12, c.SIDEWALK_HEIGHT_M))
    Entity(parent=ground_parent, model='cube', color=color.rgb(*c.SIDEWALK_COLOR),
           position=(W / 2, 0.0, H - c.SIDEWALK_HEIGHT_M / 2), scale=(W, 0.12, c.SIDEWALK_HEIGHT_M))
    # 中央の破線
    center_z = H / 2
    n_dashes = 15
    for i in range(n_dashes):
        x = (i + 0.5) * (W / n_dashes)
        Entity(parent=ground_parent, model='cube', color=color.rgb(255, 255, 100),
               position=(x, 0.01, center_z), scale=(W / n_dashes * 0.5, 0.02, 0.25))
    # 街灯・標識ポール
    for (ox, oy, orad) in c.OBSTACLES:
        pole = Entity(parent=ground_parent,
                      model=Cylinder(resolution=8, radius=orad, height=1.8),
                      color=color.rgb(90, 90, 90), position=(ox, 0.0, oy))
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
    parser.add_argument('--spawn_rate', type=float, default=0.9)
    parser.add_argument('--speed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max_agents', type=int, default=22, help='同時に存在できる歩行者数の上限（渋滞防止）')
    parser.add_argument('--trace', type=str, default=None,
                         help='指定したパスに毎フレームの各エージェント状態(位置・速度・向き等)をCSVで記録するデバッグ用オプション')
    args = parser.parse_args()

    core = SimulationCore(p_phone=args.p_phone, spawn_rate=args.spawn_rate, seed=args.seed,
                           max_agents=args.max_agents, trace_path=args.trace)
    speed_multiplier = args.speed

    app = Ursina(title="歩きスマホ行動シミュレーション（3D）", borderless=False)
    window.color = color.rgb(15, 18, 22)

    build_scene(core)

    AmbientLight(color=color.rgba(140, 140, 150, 255))
    sun = DirectionalLight()
    sun.look_at(Vec3(1, -2, 1))

    W, H = core.config.WIDTH_M, core.config.TOTAL_HEIGHT_M
    cam_target = Vec3(W / 2, 0, H / 2)

    views = {}  # agent.id -> AgentView
    markers = []
    # 頭の中心(head位置1.25)よりわずかに下＝実際の目の高さに近い位置。
    # 以前は1.5(頭のてっぺんより上)になっており、「頭上から見下ろす」ような
    # 視点になってしまっていた。
    EYE_HEIGHT = 1.2
    ORBIT_FOV = camera.fov  # 俯瞰視点のデフォルトFOVを後で戻すために保持
    # デフォルトFOV(40度)は人の視野よりかなり狭く、近くの電柱などが画面の
    # 大半を覆って他が見えなくなってしまうため、一人称時だけ広角にする。
    FP_FOV = 90
    state = {
        "paused": False, "fp": False, "fp_id": None, "fp_hidden_id": None,
        "fp_cam_pos": None, "fp_cam_yaw": 0.0,
    }

    def set_body_visible(view, visible):
        view.body.visible = visible
        view.head.visible = visible
        if view.is_phone and hasattr(view, 'phone'):
            view.phone.visible = visible

    def get_agent_by_id(aid):
        for a in core.agents:
            if a.id == aid:
                return a
        return None

    def in_map_bounds(agent):
        # エージェントは画面端から最大6m(despawn猶予ゾーン)外側まで存在できるが、
        # そこは道路・歩道が描画されていない「何もない場所」なので、一人称視点の
        # 対象には選ばない。
        return 0.0 <= agent.pos[0] <= W and 0.0 <= agent.pos[1] <= H

    def visible_agent_ids():
        return [a.id for a in core.agents if in_map_bounds(a)]

    def enter_first_person(target_id=None):
        ids = visible_agent_ids()
        if not ids:
            return
        state["fp"] = True
        state["fp_id"] = target_id if target_id in ids else ids[0]
        camera.fov = FP_FOV

    def exit_first_person():
        prev = state["fp_hidden_id"]
        if prev is not None and prev in views:
            set_body_visible(views[prev], True)
        state["fp"] = False
        state["fp_id"] = None
        state["fp_hidden_id"] = None
        camera.fov = ORBIT_FOV
        apply_camera()

    def next_first_person_target():
        if not state["fp"]:
            return
        ids = visible_agent_ids()
        if not ids:
            return
        if state["fp_id"] in ids:
            idx = ids.index(state["fp_id"])
            state["fp_id"] = ids[(idx + 1) % len(ids)]
        else:
            state["fp_id"] = ids[0]

    jp_font = ensure_japanese_font()
    hud_kwargs = {"font": jp_font} if jp_font else {}
    hud = Text(text="", position=(-0.86, 0.46), scale=1.1, background=True, **hud_kwargs)

    def refresh_hud():
        if state["fp"]:
            view = views.get(state["fp_id"])
            who = "スマホ使用者" if (view and view.is_phone) else "通常歩行者"
            view_line = f"視点: 一人称 (ID {state['fp_id']}, {who}) [N:次の人 F:視点解除]"
        else:
            view_line = "視点: 俯瞰 [クリック:その人の視点に F:誰かの視点に入る ←→:回転 ↑↓:見下ろす角度 +/-:ズーム]"
        hud.text = (
            f"Time: {core.sim_time:.1f}s\n"
            f"通過: {len(core.completed_agents)}  衝突: {core.total_collisions}  柱接触: {core.total_obstacle_collisions}\n"
            f"よろめき: {core.total_stumbles}  ニアミス: {core.total_near_misses}\n"
            f"{view_line}\n"
            f"[Space:一時停止 Q:終了]"
        )

    # 低すぎる/近すぎる角度だと歩道の横幅(WIDTH_M)が画角に収まらず、
    # 端で発生したエージェントの大半が視界外になってしまうため、
    # 全体を見渡せる構図をデフォルトにする（矢印キー/ズームで調整可）
    cam_state = {"yaw": 0.0, "pitch": 55.0, "dist": 28.0}

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

        # 簡易オービットカメラ（矢印キーで回転・上下、+/-でズーム）。一人称視点中は無効。
        if not state["fp"]:
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

            if state["fp"]:
                current = get_agent_by_id(state["fp_id"])
                if current is None or not in_map_bounds(current):
                    next_first_person_target()
                current = get_agent_by_id(state["fp_id"])
                if current is None or not in_map_bounds(current) or state["fp_id"] not in views:
                    exit_first_person()

            if state["fp"]:
                view = views[state["fp_id"]]
                agent_obj = get_agent_by_id(state["fp_id"])
                target_pos = Vec3(agent_obj.pos[0], EYE_HEIGHT, agent_obj.pos[1]) if agent_obj else view.head.world_position
                # view.root.rotation_y は update_from() で既に YAW_FREEZE_MIN_SPEED
                # による凍結処理を通った値なので、ここでは再計算せずそのまま使う
                # （別のしきい値で再計算すると凍結ロジックと矛盾してしまうため）。
                target_yaw = view.root.rotation_y

                if state["fp_hidden_id"] != state["fp_id"]:
                    # 対象切り替え直後は補間せず、即座に新しい対象の視点へスナップする
                    prev = state["fp_hidden_id"]
                    if prev is not None and prev in views:
                        set_body_visible(views[prev], True)
                    set_body_visible(view, False)
                    state["fp_hidden_id"] = state["fp_id"]
                    state["fp_cam_pos"] = target_pos
                    state["fp_cam_yaw"] = target_yaw

                view.fov.enabled = False
                view.lapse_dot.enabled = False

                # 歩行ボブや速度ベクトルの微小な揺れをそのまま反映すると酔いやすいため、
                # 位置・向きを滑らかに追従させる（位置は素早く、向きはややゆっくり、
                # かつ1フレームあたりの最大回頭角を制限して急な方向転換を和らげる）。
                pos_alpha = min(1.0, 12.0 * ursina_time.dt)
                prev_pos = state["fp_cam_pos"]
                state["fp_cam_pos"] = Vec3(
                    prev_pos.x + (target_pos.x - prev_pos.x) * pos_alpha,
                    prev_pos.y + (target_pos.y - prev_pos.y) * pos_alpha,
                    prev_pos.z + (target_pos.z - prev_pos.z) * pos_alpha,
                )

                yaw_alpha = min(1.0, 5.0 * ursina_time.dt)
                desired_yaw = lerp_angle_deg(state["fp_cam_yaw"], target_yaw, yaw_alpha)
                max_turn = 150.0 * ursina_time.dt
                turn = (desired_yaw - state["fp_cam_yaw"] + 180) % 360 - 180
                turn = max(-max_turn, min(max_turn, turn))
                state["fp_cam_yaw"] += turn

                camera.position = state["fp_cam_pos"]
                camera.rotation = Vec3(0, state["fp_cam_yaw"], 0)

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
        elif key == 'f':
            if state["fp"]:
                exit_first_person()
            else:
                enter_first_person()
        elif key == 'n':
            next_first_person_target()
        elif key == 'left mouse down' and not state["fp"]:
            hovered = mouse.hovered_entity
            aid = getattr(hovered, 'agent_id', None)
            if aid is not None:
                enter_first_person(aid)

    apply_camera()
    controller = Entity()
    controller.update = update
    controller.input = input

    try:
        app.run()
    finally:
        core.close_trace()


if __name__ == '__main__':
    main()
