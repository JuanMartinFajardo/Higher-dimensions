import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
from surfaces import TopologicalSurface, createTorus, createMoebiusStrip, createKleinBottle, createProjectivePlane, createMoebiusStrip2



# --- Motor del Juego (Pygame + PyOpenGL) ---

class TopologyGameEngine:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        pygame.init()
        pygame.font.init()
        
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Motor Topológico (Python/Pygame/OpenGL)")
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1) # --- NUEVO --- Luz de relleno
        glEnable(GL_COLOR_MATERIAL)
        glShadeModel(GL_SMOOTH)
        glLightfv(GL_LIGHT0, GL_POSITION, [5, 5, 5, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.6, 0.6, 0.6, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.5, 0.5, 0.5, 1])
        glLightfv(GL_LIGHT1, GL_POSITION, [-5, -5, -2, 1]) # --- NUEVO ---
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.2, 0.2, 0.2, 1]) # --- NUEVO ---
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.2, 0.2, 0.2, 1]) # --- NUEVO ---
        
        self.font_m = pygame.font.SysFont('Arial', 18)
        self.font_s = pygame.font.SysFont('Arial', 14)
        
        self.quad = gluNewQuadric()

        # Estado del juego
        self.surface_type = 'torus'
        self.player_pos = {'u': 0.5, 'v': 0.5}
        self.view_angle = 0
        self.orientation = 1
        self.turns_completed = {'u': 0, 'v': 0}
        self.speed = 0.02
        self.view_radius = 0.3
        self.surface = createTorus()

        # Estado de entrada
        self.keys_pressed = {}
        self.mouse_down = False
        self.last_mouse_x = 0
        
        # Botones
        self.buttons = {
            'torus': {'rect': pygame.Rect(10, 50, 60, 30), 'label': 'Toro'},
            'moebius': {'rect': pygame.Rect(75, 50, 130, 30), 'label': 'Banda de Mobius'},
            'moebius2': {'rect': pygame.Rect(210, 50, 90, 30), 'label': 'Mobius 2'},
            'klein': {'rect': pygame.Rect(305, 50, 130, 30), 'label': 'Botella de Klein'},
            'projective': {'rect': pygame.Rect(440, 50, 140, 30), 'label': 'Plano Proyectivo'}
            
        }
        

        # --- NUEVO --- Cache para la malla y la métrica
        self.mesh_data = (None, None, None) # (pos, norms, idx)
        self.world_basis_vectors = (np.array([0.15,0,0]), np.array([0,0.15,0]))
        self.cached_metric = {}
        self.cached_basis = {}
        self.dirty_mesh = True 

    def set_surface(self, new_type):
        self.surface_type = new_type
        if new_type == 'torus': self.surface = createTorus()
        elif new_type == 'moebius': self.surface = createMoebiusStrip()
        elif new_type == 'klein': self.surface = createKleinBottle()
        elif new_type == 'projective': self.surface = createProjectivePlane()
        elif new_type == 'moebius2': self.surface = createMoebiusStrip2()
        
        self.player_pos = {'u': 0.1, 'v': 0 if new_type == 'moebius' else 0.1}
        self.orientation = 1
        self.turns_completed = {'u': 0, 'v': 0}
        self.player_local_offset = np.array([0.0, 0.0], dtype=np.float32) # Reset offset
        self.dirty_mesh = True



    # --- NUEVO --- Helper para calcular los ejes del MUNDO (Naranja/Cian)
# --- MODIFICADO (Problema 2) ---
    # Los ejes del MUNDO ahora son fijos y siempre dextrógiros (R-handed).
    # Ya no dependen de la orientación del jugador.
# Los ejes del MUNDO vuelven a depender de la orientación del jugador
    def calculate_world_basis_vectors(self, orientation):
        try:
            g = self.cached_metric
            basis = self.cached_basis
            g11_sqrt = math.sqrt(g['g11'])
            g22_sqrt = math.sqrt(g['g22'])

            vec_u_x = basis['e1'][0] * g11_sqrt
            vec_u_y = basis['e1'][1] * g11_sqrt
            v_u_3d = np.array([vec_u_x, vec_u_y, 0])
            
            vec_v_x = basis['e2'][0] * g22_sqrt
            vec_v_y = basis['e2'][1] * g22_sqrt
            v_v_3d = np.array([vec_v_x, vec_v_y, 0])

            # Normalizar y escalar
            norm_u = np.linalg.norm(v_u_3d)
            norm_v = np.linalg.norm(v_v_3d)
            if norm_u > 0: v_u_3d = v_u_3d / norm_u * 0.15
            if norm_v > 0: v_v_3d = v_v_3d / norm_v * 0.15
            
            # --- RE-AÑADIDO ---
            # Si la orientación se invierte, invertimos el eje V
            if orientation < 0:
                v_v_3d = -v_v_3d
                
            return v_u_3d, v_v_3d
        except Exception:
            return (np.array([0.15,0,0]), np.array([0,0.15,0]))

    # --- NUEVO --- Helper para proyectar un punto UV a R3 local
    def project_point_to_R3(self, u, v, centerU, centerV):
        try:
            du = u - centerU
            dv = v - centerV
            
            basis = self.cached_basis
            g = self.cached_metric
            g11_sqrt = math.sqrt(g['g11'])
            g22_sqrt = math.sqrt(g['g22'])
            
            x = basis['e1'][0] * du * g11_sqrt + basis['e2'][0] * dv * g22_sqrt
            y = basis['e1'][1] * du * g11_sqrt + basis['e2'][1] * dv * g22_sqrt
            
            K = self.surface.getGaussianCurvature(centerU, centerV)
            z = -K * (x*x + y*y) * 0.5
            
            return np.array([x, y, z])
        except Exception:
            return np.array([0, 0, 0])

    # --- MODIFICADO --- movePlayer ahora solo actualiza el estado UV
    def movePlayer(self, du, dv):
        newU = self.player_pos['u'] + du
        newV = self.player_pos['v'] + dv
        
        orientation_changed = False

        if self.surface.wrapU:
            if newU >= 1:
                self.turns_completed['u'] += 1; newU -= 1
                if self.surface.orientationFlipU: self.orientation *= -1; orientation_changed = True
            elif newU < 0:
                self.turns_completed['u'] -= 1; newU += 1
                if self.surface.orientationFlipU: self.orientation *= -1; orientation_changed = True
        else:
            newU = max(0, min(1, newU))
        
        if self.surface.wrapV:
            if newV >= 1:
                self.turns_completed['v'] += 1; newV -= 1
                if self.surface.orientationFlipV: self.orientation *= -1; orientation_changed = True
            elif newV < 0:
                self.turns_completed['v'] -= 1; newV += 1
                if self.surface.orientationFlipV: self.orientation *= -1; orientation_changed = True
        else:
            v_range = -0.3 if self.surface.name == "Banda de Möbius" else 0
            newV = max(v_range, min(abs(v_range), newV)) if v_range < 0 else max(0, min(1, newV))

        self.player_pos = {'u': newU, 'v': newV}
        self.dirty_mesh = True # Forzar recálculo de malla en el *próximo* frame
        
    def handle_input(self):
        running = True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                self.keys_pressed[event.key] = True
            elif event.type == pygame.KEYUP:
                self.keys_pressed[event.key] = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    clicked_button = False
                    for name, data in self.buttons.items():
                        if data['rect'].collidepoint(event.pos):
                            self.set_surface(name)
                            clicked_button = True
                            break
                    if not clicked_button:
                        self.mouse_down = True
                        self.last_mouse_x = event.pos[0]
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_down:
                    deltaX = event.pos[0] - self.last_mouse_x
                    self.view_angle += deltaX * 0.01
                    self.last_mouse_x = event.pos[0]
        
        # --- MODIFICADO (Problema 1) ---
        # Rotación (cámara) - suave
        if self.keys_pressed.get(pygame.K_LEFT): self.view_angle += 0.05
        if self.keys_pressed.get(pygame.K_RIGHT): self.view_angle -= 0.05               
        # --- MODIFICADO ---
        # Movimiento (mundo) - vuelve a la lógica original
        forward, right = 0, 0
        if self.keys_pressed.get(pygame.K_w): forward += 1
        if self.keys_pressed.get(pygame.K_s): forward -= 1
        if self.keys_pressed.get(pygame.K_a): right -= 1
        if self.keys_pressed.get(pygame.K_d): right += 1
        
        if forward != 0 or right != 0:
            cos = math.cos(self.view_angle)
            sin = math.sin(self.view_angle)
            # Calculamos el delta UV y movemos al jugador
            du = (right * cos - forward * sin) * self.speed
            dv = (right * sin + forward * cos) * self.speed
            self.movePlayer(du, dv) # Llama a movePlayer directamente
            
        return running


    # --- NUEVO (Problema 2) --- Helper para dibujar flechas 3D
    def draw_3d_arrow(self, vector, color, radius=0.015):
        length = np.linalg.norm(vector)
        if length < 1e-6: return

        glPushMatrix()
        glColor3fv(color)
        
        v_norm = vector / length
        z_axis = np.array([0, 0, 1])
        angle_rad = math.acos(np.dot(z_axis, v_norm))
        angle_deg = math.degrees(angle_rad)
        axis = np.cross(z_axis, v_norm)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-6: # Paralelo o anti-paralelo
            if v_norm[2] < 0: # Apuntando hacia abajo
                glRotatef(180, 1, 0, 0)
        else:
            axis = axis / axis_norm
            glRotatef(angle_deg, axis[0], axis[1], axis[2])

        # Cuerpo del cilindro
        gluCylinder(self.quad, radius, radius, length * 0.8, 8, 1)
        # Cabeza (cono)
        glTranslatef(0, 0, length * 0.8)
        gluCylinder(self.quad, radius * 2.5, 0.0, length * 0.2, 8, 1)
        
        glPopMatrix()

    def draw_3d(self):
        # --- MODIFICADO --- Recalcular la malla SÓLO si es necesario
        if self.dirty_mesh or self.mesh_data[0] is None:
            pos, norms, idx = self.surface.renderLocalMesh(
                self.player_pos['u'], self.player_pos['v'], self.view_radius, self.orientation
            )
            self.mesh_data = (pos, norms, idx)
            
            # Cachear la métrica y base en esta posición
            self.cached_metric = self.surface.getMetric(self.player_pos['u'], self.player_pos['v'])
            self.cached_basis = self.surface.getTangentBasis(self.player_pos['u'], self.player_pos['v'])
            
            # Calcular los ejes del MUNDO (Naranja/Cian) usando la orientación actual
            self.world_basis_vectors = self.calculate_world_basis_vectors(self.orientation)
            
            self.dirty_mesh = False
        
        pos, norms, idx = self.mesh_data
        v_u_3d, v_v_3d = self.world_basis_vectors
        
        # --- Configuración de Cámara ---
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(75, (self.width / self.height), 0.01, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        cos_a = math.cos(self.view_angle)
        sin_a = math.sin(self.view_angle)
        cam_x, cam_y, cam_z = sin_a * 1.2, -cos_a * 1.2, 0.6
        gluLookAt(cam_x, cam_y, cam_z, 0, 0, 0, 0, 0, 1)

        glClearColor(0.53, 0.81, 0.92, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        # --- DIBUJAR JUGADOR Y SUS EJES (Rojo/Azul) ---
        # (Se dibujan en (0,0,0), *antes* de mover el mundo)
        glEnable(GL_LIGHTING)
        glColor3f(1.0, 0.2, 0.4) # (0xff3366)
        gluSphere(self.quad, 0.05, 16, 16)

        glDisable(GL_LIGHTING)
        # Ejes del JUGADOR (Rojo/Azul) - relativos a la cámara
        forward_vec = np.array([-sin_a, cos_a, 0]) * 0.2
        right_vec = np.array([cos_a, sin_a, 0]) * 0.2
        self.draw_3d_arrow(forward_vec, (1, 0, 0)) # Adelante (Rojo)
        self.draw_3d_arrow(right_vec, (0, 0, 1))   # Derecha (Azul)
        

        # --- DIBUJAR EL MUNDO (Malla y Landmarks) ---
        glEnable(GL_LIGHTING)
        
        # 1. Renderizar la superficie (usando datos cacheados)
        if pos is not None and len(idx) > 0:
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            
            glVertexPointer(3, GL_FLOAT, 0, pos)
            glNormalPointer(GL_FLOAT, 0, norms)
            
            # Malla sólida
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            glMaterialfv(GL_FRONT, GL_SPECULAR, [0.3, 0.3, 0.3, 1])
            glMaterialf(GL_FRONT, GL_SHININESS, 30.0)
            glColor3f(0.29, 0.56, 0.89) # (0x4a90e2)
            glDrawElements(GL_TRIANGLES, len(idx), GL_UNSIGNED_INT, idx)
            
            # Wireframe
            glDisable(GL_LIGHTING)
            glPolygonOffset(-1.0, -1.0)
            glEnable(GL_POLYGON_OFFSET_LINE)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glColor4f(0, 0, 0, 0.2)
            glDrawElements(GL_TRIANGLES, len(idx), GL_UNSIGNED_INT, idx)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDisable(GL_POLYGON_OFFSET_LINE)
            glEnable(GL_LIGHTING)

            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            
        # 2. Renderizar landmarks (CON EJES DEL MUNDO)
        for lm in self.surface.landmarks:
            dist = self.surface.uvDistance(self.player_pos['u'], self.player_pos['v'], lm['u'], lm['v'])
            # El radio de visión del landmark DEBE ser <= al radio de la malla (0.3)
            if dist < self.view_radius:
                lu, lv = self.surface.adjustForWrapping(lm['u'], lm['v'], self.player_pos['u'], self.player_pos['v'])
                
                # Proyectar la posición del landmark al espacio R3 local
                lm_pos = self.project_point_to_R3(lu, lv, self.player_pos['u'], self.player_pos['v'])
                
                glPushMatrix()
                glTranslatef(lm_pos[0], lm_pos[1], lm_pos[2])
                
                # Esfera del landmark
                glEnable(GL_LIGHTING)
                glColor3fv(lm['color'])
                gluSphere(self.quad, 0.04, 12, 12)
                
                # Ejes del MUNDO (Naranja/Cian) - (Problema 3)
                glDisable(GL_LIGHTING)
                self.draw_3d_arrow(v_u_3d, (1.0, 0.5, 0.0)) # Eje U (Naranja)
                self.draw_3d_arrow(v_v_3d, (0.0, 1.0, 1.0)) # Eje V (Cian)
                
                glPopMatrix()

    def draw_2d(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.width, self.height, 0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        def draw_text(text, x, y, font, color=(255, 255, 255)):
            text_surface = font.render(text, True, color)
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glWindowPos2d(x, self.height - y - text_surface.get_height() + 5)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                         GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        draw_text("Motor Topológico", 10, 10, self.font_m)
        draw_text("WASD: mover | Flechas/Mouse: rotar", 10, 30, self.font_s)

        for name, data in self.buttons.items():
            rect, label = data['rect'], data['label']
            color = (0.2, 0.4, 0.8) if self.surface_type == name else (0.3, 0.3, 0.3)
            glColor3fv(color)
            glRectf(rect.left, rect.top, rect.right, rect.bottom)
            text_y_pygame = rect.top + (rect.height // 2) - 8
            draw_text(label, rect.x + 10, text_y_pygame, self.font_s)

        info_y = 100
        draw_text(f"UV: ({self.player_pos['u']:.3f}, {self.player_pos['v']:.3f}) | "
                  f"Vueltas: U={self.turns_completed['u']}, V={self.turns_completed['v']}",
                  10, info_y, self.font_s, (220, 220, 220))
        
        orient_text = f"Orientación: {'Normal (+)' if self.orientation > 0 else 'Invertida (-)'}"
        orient_color = (0, 255, 0) if self.orientation > 0 else (255, 0, 0)
        draw_text(orient_text, 10, info_y + 20, self.font_m, orient_color)


    def run(self):
        running = True
        clock = pygame.time.Clock()
        
        while running:
            # 1. Manejar entradas (solo actualiza offsets y ángulos)
            running = self.handle_input()
            
            # 2. Dibujar 3D (recalcula malla SÓLO si dirty_mesh == True)
            self.draw_3d()
            
            # 3. Dibujar UI 2D
            self.draw_2d()
            
            # 4. Actualizar pantalla
            pygame.display.flip()
            
            # 5. Esperar
            clock.tick(60)
            
        pygame.quit()

# --- Punto de entrada principal ---
if __name__ == "__main__":
    engine = TopologyGameEngine(800, 750) 
    engine.run()