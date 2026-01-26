import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

from rime.cube import StickerCube, CubeBase
from rime.cubedraw import BaseCubeRenderer


class OpenGLCubeRenderer(BaseCubeRenderer):
    COLOR_MAP = {
        'W': (1.0, 1.0, 1.0),
        'Y': (1.0, 0.84, 0.0),
        'R': (0.7, 0.0, 0.0),
        'O': (1.0, 0.4, 0.0),
        'G': (0.0, 0.55, 0.25),
        'B': (0.0, 0.3, 0.8),
    }

    def __init__(self, cube, width=1000, height=1000,):
        super().__init__(cube, scale= (2.8 / cube.n))  # n * sqrt(3)
        self.width = width
        self.height = height

        self.init_gl()

    def init_gl(self):
        glViewport(0, 0, self.width, self.height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 50.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -8.0)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClearColor(0.12, 0.12, 0.12, 1.0)

    def resize(self, width, height):
        self.width = width
        self.height = height

        glViewport(0, 0, width, height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height if height else 1.0, 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -6.0)

    def zoom(self, delta):
        """
        delta > 0: 放大
        delta < 0: 缩小
        """
        factor = 1.1 if delta > 0 else 0.9
        self.scale *= factor

        # 防止缩到看不见 / 炸屏
        self.scale = max(self.scale, 0.05)
        self.scale = min(self.scale, 5.0)

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPushMatrix()

        glScalef(self.scale, self.scale, self.scale)

        # 全局观察旋转
        glRotatef(np.degrees(self.angles[0]), 1, 0, 0)
        glRotatef(np.degrees(self.angles[1]), 0, 1, 0)
        glRotatef(np.degrees(self.angles[2]), 0, 0, 1)

        self.colors = self.cube.color
        for face, i, j, quad in self.compute_face_quads():
            q = np.array(quad)
            color = self.colors[face][i][j]

            is_ghost = False
            # partial rotation
            if self.partial is not None:
                axis, layer, ang = self.partial
                if CubeBase.should_rotate_by_sticker(face, i, j, axis, layer, self.n):
                    layer_geom = CubeBase.layer_to_geom(layer, self.n)
                    q = CubeBase.rotate_around_layer(q, axis, layer_geom, ang)
                    is_ghost = True

            self.draw_face_quad(q, color, ghost=is_ghost)

        glPopMatrix()

    @classmethod
    def draw_face_quad(cls, quad, color, ghost=False):
        r, g, b = cls.COLOR_MAP[color]
        ghost_alpha = 0.35 if ghost else 1.0
        glColor4f(r, g, b, ghost_alpha)

        glBegin(GL_QUADS)
        for x, y, z in quad:
            glVertex3f(x, y, z)
        glEnd()

        # 边框：ghost 时更亮一点
        if ghost:
            glColor4f(0.8, 0.9, 1.0, 0.9)
            glLineWidth(2.5)
        else:
            glColor4f(0.1, 0.1, 0.1, 1.0)
            glLineWidth(1.5)

        glBegin(GL_LINE_LOOP)
        for x, y, z in quad:
            glVertex3f(x, y, z)
        glEnd()

    @classmethod
    def toggle_fullscreen(cls, is_fullscreen):
        if not is_fullscreen:
            info = pygame.display.Info()
            size = (info.current_w, info.current_h)
            flags = pygame.DOUBLEBUF | pygame.OPENGL | pygame.NOFRAME
        else:
            size = (cls.WIDTH, cls.HEIGHT)
            flags = pygame.DOUBLEBUF | pygame.OPENGL

        screen = pygame.display.set_mode(size, flags)
        return screen, size

    @classmethod
    def run(cls, cube):
        pygame.init()
        pygame.display.set_mode(
            (cls.WIDTH, cls.HEIGHT),
            pygame.DOUBLEBUF | pygame.OPENGL
        )
        pygame.display.set_caption("Rubik Cube - OpenGL")

        renderer = OpenGLCubeRenderer(cube, cls.WIDTH, cls.HEIGHT)
        clock = pygame.time.Clock()

        running = True
        fullscreen = False
        while running:
            dt = clock.tick(60) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False

                    elif ev.key == pygame.K_f:
                        # toggle fullscreen
                        fullscreen = not fullscreen
                        screen, (w, h) = cls.toggle_fullscreen(fullscreen)
                        renderer.resize(w, h)

                elif ev.type == pygame.MOUSEWHEEL:
                    renderer.zoom(ev.y)

            renderer.angles[1] += 0.3 * dt
            renderer.draw()
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    cube = StickerCube(n=7)
    OpenGLCubeRenderer.run(cube)
