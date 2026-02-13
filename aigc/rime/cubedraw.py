import numpy as np  # 导入 NumPy 库，用于数值计算和处理多维数组
import pygame  # 导入 Pygame 库，用于游戏开发和图形界面设计
import math
from rime.cube import StickerCube, CubeBase
from rime.cubie import CubieBase


class CubeDraw:
    """
    表示一个立方体。
    """
    # 定义屏幕的宽度和高度
    WIDTH = 800
    HEIGHT = 800

    # 定义颜色常量
    BLACK = (0, 0, 0)  # 黑色
    WHITE = (255, 255, 255)  # 白色

    def __init__(self, pos: np.ndarray, a: float) -> None:
        """
        初始化立方体。
        :param pos: 立方体的中心位置，是一个包含三个元素的 NumPy 数组。
        :param a: 立方体的边长。
        """
        self.pos = pos  # 立方体的中心位置
        self.angle = np.pi / 4  # 立方体的旋转角度，初始化为 45 度
        self.center_offset = np.array([-a / 2, -a / 2, -a / 2])  # 立方体顶点到中心的偏移量
        self.edges = np.array([  # 立方体的边，是一个包含 12 条边的数组
            # 前脸的四条边
            np.array([np.array([0, 0, 0]), np.array([a, 0, 0])]),
            np.array([np.array([a, 0, 0]), np.array([a, a, 0])]),
            np.array([np.array([a, a, 0]), np.array([0, a, 0])]),
            np.array([np.array([0, a, 0]), np.array([0, 0, 0])]),
            # 右脸的四条边
            np.array([np.array([0, 0, 0]), np.array([0, 0, a])]),
            np.array([np.array([a, a, 0]), np.array([a, a, a])]),
            np.array([np.array([a, 0, 0]), np.array([a, 0, a])]),
            np.array([np.array([0, a, 0]), np.array([0, a, a])]),
            # 上脸的四条边
            np.array([np.array([0, 0, a]), np.array([a, 0, a])]),
            np.array([np.array([a, 0, a]), np.array([a, a, a])]),
            np.array([np.array([a, a, a]), np.array([0, a, a])]),
            np.array([np.array([0, a, a]), np.array([0, 0, a])]),
        ])

    def draw(self, screen: pygame.surface.Surface, rotation_rate: float) -> None:
        """
        在屏幕上绘制立方体。
        :param screen: 要绘制立方体的 Pygame 屏幕对象。
        :param rotation_rate: 立方体的旋转速率，用于控制立方体旋转的速度。
        """
        # 将立方体的边加上中心偏移量，得到实际的顶点位置
        rotated_cube = np.add(self.edges, self.center_offset)

        # 计算绕 X、Y、Z 轴旋转的矩阵
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(self.angle), -np.sin(self.angle)],
            [0, np.sin(self.angle), np.cos(self.angle)]
        ])
        rotation_matrix_y = np.array([
            [np.cos(self.angle), 0, np.sin(self.angle)],
            [0, 1, 0],
            [-np.sin(self.angle), 0, np.cos(self.angle)]
        ])
        rotation_matrix_z = np.array([
            [np.cos(self.angle), -np.sin(self.angle), 0],
            [np.sin(self.angle), np.cos(self.angle), 0],
            [0, 0, 1],
        ])

        # 对立方体进行旋转
        rotated_cube = np.matmul(rotated_cube, rotation_matrix_x)
        rotated_cube = np.matmul(rotated_cube, rotation_matrix_y)
        rotated_cube = np.matmul(rotated_cube, rotation_matrix_z)

        # 将旋转后的立方体移动到正确的位置
        moved_cube = np.add(self.pos, rotated_cube)

        # 在屏幕上绘制立方体的边
        for edge in moved_cube:
            # 获取边的两个端点的屏幕坐标
            start_pos = edge[0][0:2]
            end_pos = edge[1][0:2]
            # 绘制边
            pygame.draw.line(screen, self.WHITE, start_pos, end_pos)

        # 更新立方体的旋转角度
        self.angle += rotation_rate

    @classmethod
    def main(cls, a: float = 200):
        """
        主函数，启动 Pygame 并创建旋转的立方体。
        """
        # 初始化 Pygame
        pygame.init()
        # 创建屏幕对象
        screen = pygame.display.set_mode((cls.WIDTH, cls.HEIGHT))
        # 设置窗口标题
        pygame.display.set_caption("旋转立方体 By stormsha")
        # 创建立方体对象，中心位于 (400, 400, 200)，边长为 200
        cube = cls(np.array([2 * a, 2 * a, a]), a)

        # 主循环
        running = True
        while running:
            # 处理 Pygame 事件，如关闭窗口等
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 清空屏幕
            screen.fill(cls.BLACK)
            # 绘制立方体
            cube.draw(screen, 0.001)
            # 更新屏幕
            pygame.display.flip()


class BaseCubeRenderer:
    WIDTH = 1000
    HEIGHT = 1000

    def __init__(self, cube: StickerCube, scale: float = None):
        self.cube = cube
        self.n = cube.n
        self.colors: dict = cube.color
        self.partial = None
        self.last_rotated = None

        self.scale = scale
        self.angles = np.array([0.8, 0.6, 0.2])  # 初始旋转角

    def apply_partial_rotation(self, axis, layer, angle):
        self.partial = (axis, layer, angle)
        self.last_rotated = (axis, layer)

    def commit_partial(self):
        self.partial = None

    def compute_face_quads(self):
        # similar to previous helper: for each sticker produce its quad in model space
        # list of (face, i, j, quad_points)
        for face in CubieBase.FACES:
            quads = CubeBase.face_quads(face, self.n)
            for idx, quad in enumerate(quads):
                i = idx // self.n
                j = idx % self.n
                yield face, i, j, quad

    def draw(self):
        raise NotImplementedError


class CubeRenderer(BaseCubeRenderer):
    BLACK = (0, 0, 0)  # 黑色
    EDGE_COLOR = (20, 20, 20)
    HIGHLIGHT_COLOR = (0, 255, 200)
    BG_COLOR = (30, 30, 30)
    COLOR_MAP = {
        'W': (255, 255, 255),  # White
        'Y': (255, 213, 0),  # Yellow
        'R': (180, 0, 0),  # Red
        'O': (255, 100, 0),  # Orange
        'G': (0, 140, 60),  # Green
        'B': (0, 70, 200),  # Blue
    }

    # COLOR_MAP = {
    #     'W': (255, 255, 255),  # 白
    #     'Y': (255, 255, 0),  # 黄
    #     'R': (255, 0, 0),  # 红
    #     'O': (255, 128, 0),  # 橙
    #     'G': (0, 200, 0),  # 绿
    #     'B': (0, 0, 255),  # 蓝
    # }

    def __init__(self, cube: "StickerCube", screen=None):
        super().__init__(cube, scale=min(self.WIDTH, self.HEIGHT) * 0.45 / cube.n)  # 每个小方块大小缩放

        self.offset = -self.n / 2  # 三维坐标范围 [-n/2, +n/2]
        self.center = (self.WIDTH // 2, self.HEIGHT // 2)
        self.window_size = (self.WIDTH, self.HEIGHT)
        self.screen = screen or pygame.display.set_mode(self.window_size)

    # --- 3D 转 2D ---
    def project(self, p3):
        """正交投影，中心偏移到屏幕中央"""
        x, y, z = p3  # point_3d
        sx = int(self.center[0] + x * self.scale)
        sy = int(self.center[1] - y * self.scale)
        return sx, sy

    # --- 绘制一个贴纸平面 (quad) ---
    def draw_face_quad(self, quad: np.ndarray, color: str):
        pts = [self.project(p) for p in quad]
        pygame.draw.polygon(self.screen, self.COLOR_MAP[color], pts)
        pygame.draw.polygon(self.screen, self.EDGE_COLOR, pts, 2)  # 黑色边框

    # --- 绘制整个魔方 ---
    def draw_cube(self):
        R = CubeBase.rotation_matrix(self.angles)

        # 根据 z 排序（从近到远）
        all_quads = []
        for face, i, j, quad in self.compute_face_quads():
            q = np.array([R @ v for v in quad])
            z_avg = np.mean(q[:, 2])
            fcolor = self.colors[face]  # Face->二维颜色
            all_quads.append((z_avg, q, fcolor[i][j]))

        # 从远到近绘制
        all_quads.sort()  # z 从小到大

        for _, quad, color in all_quads:
            self.draw_face_quad(quad, color)

    def draw_net(self, x, y, size: float):
        """
        simple unfolded net layout: U on top, L F R B in middle, D bottom
        魔方展开图渲染：
        - U 在顶部
        - L F R B 在中间
        - D 在底部
            U
        L   F   R   B
            D
        """
        n = self.n
        sticker = int(max(size // (4 * n), 3))  # 一行有 4 个面（L F R B）
        # net positions for faces，标准魔方 net 坐标
        layout = {
            'U': (1, 0), 'L': (0, 1), 'F': (1, 1), 'R': (2, 1), 'B': (3, 1), 'D': (1, 2)
        }
        # self.colors = self.cube.color
        for face, (cx, cy) in layout.items():
            fx = x + cx * sticker * n
            fy = y + cy * sticker * n
            for i in range(n):
                for j in range(n):
                    rx = fx + j * sticker
                    ry = fy + i * sticker
                    ii = n - 1 - i  # 坐标系契约对齐
                    jj = n - 1 - j
                    color = self.COLOR_MAP[self.colors[face][ii][jj]]
                    rect = (rx, ry, sticker - 1, sticker - 1)
                    pygame.draw.rect(self.screen, color, rect)  # 贴纸背景
                    pygame.draw.rect(self.screen, self.EDGE_COLOR, rect, 1)  # 边框

    def draw_net_auto(self):
        if self.n > 15:
            return
        net_total = int(min(self.WIDTH, self.HEIGHT) * 0.24)  # 大小配置：屏幕短边的 24%
        padding = int(min(self.WIDTH, self.HEIGHT) * 0.03)  # 边距：屏幕短边的 3%
        x = padding
        y = self.HEIGHT - net_total - padding
        self.draw_net(x, y, net_total)

    def draw(self):
        '''局部动作 ⟶ 全局观察'''
        self.screen.fill(self.BG_COLOR)

        R = CubeBase.rotation_matrix(self.angles)
        quads = self.compute_face_quads()
        self.colors = self.cube.color
        # build transformed quads and z depth, taking into account partial rotation
        draw_list = []
        for face, i, j, quad in quads:
            q = np.array(quad)
            color = self.colors[face][i][j]
            # if partial affects these sticker positions, rotate them in model space
            if self.partial is not None:
                axis, layer, ang = self.partial
                if CubeBase.should_rotate_by_sticker(face, i, j, axis, layer, self.n):
                    # old_face, new_r, new_c = CubeBase.rotated_coord(q, axis, layer, self.n, ang)
                    # color = colors[old_face][new_r][new_c]
                    layer_geom = CubeBase.layer_to_geom(layer, self.n)
                    q = CubeBase.rotate_around_layer(q, axis, layer_geom, ang)

            q2 = np.array([R @ v for v in q])
            z = q2[:, 2].mean()

            draw_list.append((z, face, q2, color))

        # painter's algorithm: draw from far to near
        draw_list.sort(key=lambda x: x[0])
        for _, face, q2, color in draw_list:
            self.draw_face_quad(q2, color=color)

        # 高亮上一次旋转的层（旋转结束后）
        # if self.last_rotated is not None and self.partial is None:
        #     axis, layer = self.last_rotated
        #     for _, face, q2, color in draw_list:
        #         # center = np.mean(q2, axis=0)
        #         # layer_coord = CubeBase.layer_to_logic(layer, self.n)
        #         if CubeBase.should_rotate(q2, axis, layer, self.n):  # np.isclose(center[axis], layer_coord)
        #             pts = [self.project(p) for p in q2]
        #             pygame.draw.polygon(self.screen, self.HIGHLIGHT_COLOR, pts, 2)

        # draw a small 2D net in the lower-left for debugging
        # self.draw_net(20, self.HEIGHT - 220, 200)

    # --- 主循环 ---
    @classmethod
    def run(cls, cube: StickerCube, rot_speed: float = 0.007):
        pygame.init()
        pygame.display.set_caption("Rubik's Cube 3D Draw")

        screen = pygame.display.set_mode((cls.WIDTH, cls.HEIGHT))
        drawer = cls(cube, screen=screen)
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            drawer.screen.fill(cls.BG_COLOR)
            drawer.draw_cube()

            # 自动旋转
            drawer.angles += rot_speed  # 旋转速度

            pygame.display.flip()
            clock.tick(60)


class RotationAnimation:
    def __init__(self, axis: int, layer: int, direction: int, duration: float = 0.25):
        self.op = (axis, layer, direction)
        self.duration = duration
        self.elapsed = 0

    def step(self, dt):
        self.elapsed += dt
        t = min(self.elapsed / self.duration, 1.0)
        angle = 90 * t * self.op[2]
        return (t >= 1.0), angle


class RubiksCubeDraw:
    ROT_DURATION = 0.22  # seconds per 90deg

    def __init__(self, cube: StickerCube = None, auto_rotate: bool = True):
        self.renderer = CubeRenderer(cube or StickerCube(3))
        self.cube = self.renderer.cube

        self.clock = pygame.time.Clock()
        self.pending = []  # list of (axis, layer, dir)
        self.current_anim = None  # 当前动画状态
        self.paused = False

        # dragging view
        self.auto_rotate = auto_rotate
        self.view_drag = False
        self.last_mouse = (0, 0)
        # face drag for layer turns (simple heuristic)
        self.face_dragging = False
        self.face_drag_start = None
        self.face_drag_face = None

    # enqueue moves (primitive moves)
    def enqueue_moves(self, moves: list):
        if not isinstance(moves, (list, tuple)):
            moves = [moves]
        self.pending.extend(moves)
        print('moves:', moves)

    def play(self):
        self.paused = False

    def pause(self):
        self.paused = True

    def update(self, dt: float):
        # auto-rotate view,每帧调用一次
        if self.auto_rotate and not self.view_drag:
            self.renderer.angles[1] += 0.2 * dt

        if self.paused:
            return

        if self.current_anim is None and self.pending:
            op = self.pending.pop(0)
            self.current_anim = RotationAnimation(*op, duration=self.ROT_DURATION)

        if self.current_anim is not None:
            done, angle = self.current_anim.step(dt)
            axis, layer, dir = self.current_anim.op
            self.renderer.apply_partial_rotation(axis, layer, math.radians(angle))
            if done:
                # commit to model
                self.cube.rotate(axis, layer, dir)
                self.renderer.commit_partial()
                self.current_anim = None

    # Mouse helpers (view drag + face drag)
    def handle_mouse_down(self, pos, button):
        if button == 1:  # left
            # start potential face drag or view drag
            self.last_mouse = pos
            self.view_drag = True  # mouse_pressed
        elif button == 3:  # right button: start face drag (heuristic)
            self.face_dragging = True
            self.face_drag_start = pos
            # pick a face under cursor using renderer projection (simplified)
            self.face_drag_face = self._pick_face_at(pos)

    def handle_mouse_up(self, pos, button):
        if button == 1:
            self.view_drag = False
        elif button == 3:
            if self.face_dragging and self.face_drag_face is not None:
                dx = pos[0] - self.face_drag_start[0]
                dy = pos[1] - self.face_drag_start[1]
                axis, layer, direction = self._infer_turn_from_drag(self.face_drag_face, dx, dy)
                self.enqueue_moves([(axis, layer, direction)])
            self.face_dragging = False
            self.face_drag_face = None
            self.face_drag_start = None

    def handle_mouse_move(self, pos):
        if self.view_drag:
            dx = pos[0] - self.last_mouse[0]
            dy = pos[1] - self.last_mouse[1]
            self.renderer.angles[1] += dx * 0.005
            self.renderer.angles[0] += dy * 0.005
            self.last_mouse = pos

    def _pick_face_at(self, pos):
        # heuristic: return nearest visible face by screen position
        # map screen pos back to face centers
        cx, cy = self.renderer.center
        rx = (pos[0] - cx) / self.renderer.scale
        ry = (cy - pos[1]) / self.renderer.scale
        # choose face by quadrant: rough mapping
        if ry > 1.2:
            return 'U'
        if ry < -1.2:
            return 'D'
        if rx > 1.2:
            return 'R'
        if rx < -1.2:
            return 'L'
        return 'F'

    def _infer_turn_from_drag(self, face: str, dx, dy):
        """
        根据鼠标拖拽推断一个 primitive 层转。
        使用 cube.face_axis 统一映射，以保持与 rotate() 方向一致。
        """
        mid = self.cube.n // 2
        # 1. cube 内置 face → (axis, side) 映射（R/L → X轴, U/D → Y轴, F/B → Z轴）
        axis, side = CubeBase.face_axis[face]
        # 2. 选择层
        layer = -mid if side == 0 else mid
        # 3. 从 dx/dy 推方向（哪边拖拽多就用那轴）
        major = dx if abs(dx) > abs(dy) else dy
        # 4. 方向统一使用 cube 的 rotate 规范：
        direction = 1 if major > 0 else -1

        return axis, layer, direction

    # main loop
    def run(self):
        pygame.init()
        pending_moves = []
        pygame.display.set_caption("Interactive Rubik's Cube (Left drag=view, Right drag=turn)")
        running = True
        cubie = CubieBase(n=self.cube.n)

        while running:
            dt = self.clock.tick(60) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_down(ev.pos, ev.button)
                elif ev.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up(ev.pos, ev.button)
                elif ev.type == pygame.MOUSEMOTION:
                    self.handle_mouse_move(ev.pos)
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_a:  # 暂停 / 恢复自动旋转
                        self.auto_rotate = not self.auto_rotate
                    elif ev.key == pygame.K_SPACE:  # 暂停 / 恢复动画队列播放
                        self.paused = not self.paused
                    elif ev.key == pygame.K_p:  # 单次
                        seq = self.cube.propose_move()
                        pending_moves = [seq]
                        self.enqueue_moves(pending_moves)
                    elif ev.key == pygame.K_i:  # 回退上一次
                        seq = self.cube.invert_moves(pending_moves)
                        self.enqueue_moves(seq)
                        pending_moves.clear()
                    elif ev.key == pygame.K_g:  # 生成并播放 scramble 序列
                        pending_moves = self.cube.generate_moves(25)
                        self.enqueue_moves(pending_moves)
                    elif ev.key == pygame.K_s:
                        pending_moves = cubie.solve_sticker(self.cube.get_state())
                        print(len(pending_moves))
                        self.enqueue_moves(pending_moves)
                    elif ev.key == pygame.K_l:
                        print(self.cube.color)
                    elif ev.key == pygame.K_c:  # 清空 pending 队列
                        # clear pending
                        self.pending.clear()
                        pending_moves.clear()
                    elif ev.key == pygame.K_r:
                        # restart: reinit model if possible
                        self.cube.reset()

            self.update(dt)
            self.renderer.draw()
            self.renderer.draw_net_auto()

            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    # 如果脚本被直接运行，则执行主函数
    # CubeDraw.main()

    cube = StickerCube(n=3)  # 初始解法状态
    # mv = cube.scramble(20)
    # cube.apply(mv)
    # CubeRenderer.run(cube)

    app = RubiksCubeDraw(cube)
    app.run()
    print(cube.color)
