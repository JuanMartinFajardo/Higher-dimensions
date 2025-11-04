import math
import numpy as np


# --- Lógica Principal de la Superficie Topológica ---

class TopologicalSurface:
    def __init__(self):
        self.triangles = []
        self.name = ""
        self.landmarks = []
        self.metric = None
        self.curvature = None
        self.wrapU = False
        self.wrapV = False
        self.orientationFlipU = False
        self.orientationFlipV = False

    def createRegularTriangulation(self, resU, resV, uRange, vRange):
        self.triangles = []
        uMin, uMax = uRange
        vMin, vMax = vRange
        
        du = (uMax - uMin) / resU
        dv = (vMax - vMin) / resV
        
        for i in range(resU):
            for j in range(resV):
                u0 = uMin + i * du
                u1 = uMin + (i + 1) * du
                v0 = vMin + j * dv
                v1 = vMin + (j + 1) * dv
                
                self.triangles.append({
                    'v0': (u0, v0),
                    'v1': (u1, v0),
                    'v2': (u0, v1)
                })
                
                self.triangles.append({
                    'v0': (u1, v0),
                    'v1': (u1, v1),
                    'v2': (u0, v1)
                })

    def normalizeUV(self, u, v):
        normU, normV = u, v
        if self.wrapU: normU = ((u % 1) + 1) % 1
        if self.wrapV: normV = ((v % 1) + 1) % 1
        return normU, normV

    def uvDistance(self, u1, v1, u2, v2):
        du = abs(u2 - u1)
        dv = abs(v2 - v1)
        if self.wrapU: du = min(du, 1 - du)
        if self.wrapV: dv = min(dv, 1 - dv)
        return math.sqrt(du*du + dv*dv)

    def getMetric(self, u, v):
        if self.metric: return self.metric(u, v)
        return {'g11': 1, 'g12': 0, 'g22': 1}

    def getTangentBasis(self, u, v):
        g = self.getMetric(u, v)
        if g['g11'] <= 0: g['g11'] = 0.0001 # Evitar sqrt(0)
        e1_len = math.sqrt(g['g11'])
        e1 = np.array([1 / e1_len, 0])
        e2_unnorm = np.array([-g['g12'], g['g11']])
        e2_len_sq = (g['g11'] * e2_unnorm[0] * e2_unnorm[0] +
                     2 * g['g12'] * e2_unnorm[0] * e2_unnorm[1] +
                     g['g22'] * e2_unnorm[1] * e2_unnorm[1])
        if e2_len_sq <= 0: return {'e1': e1, 'e2': np.array([0, 1])} # Fallback
        e2_len = math.sqrt(e2_len_sq)
        e2 = e2_unnorm * (1 / e2_len)
        return {'e1': e1, 'e2': e2}

    def adjustForWrapping(self, u, v, centerU, centerV):
        adjU, adjV = u, v
        if self.wrapU:
            options = [u - 1, u, u + 1]
            adjU = min(options, key=lambda curr: abs(curr - centerU))
        if self.wrapV:
            options = [v - 1, v, v + 1]
            adjV = min(options, key=lambda curr: abs(curr - centerV))
        return adjU, adjV

    def getGaussianCurvature(self, u, v):
        if self.curvature: return self.curvature(u, v)
        return 0

    def projectTriangleToR3(self, tri, centerU, centerV, orientation):
        u0, v0 = self.normalizeUV(tri['v0'][0], tri['v0'][1])
        u1, v1 = self.normalizeUV(tri['v1'][0], tri['v1'][1])
        u2, v2 = self.normalizeUV(tri['v2'][0], tri['v2'][1])
        
        u0, v0 = self.adjustForWrapping(u0, v0, centerU, centerV)
        u1, v1 = self.adjustForWrapping(u1, v1, centerU, centerV)
        u2, v2 = self.adjustForWrapping(u2, v2, centerU, centerV)
        
        basis = self.getTangentBasis(centerU, centerV)
        g = self.getMetric(centerU, centerV)
        
        def toLocal(du, dv):
            g11_sqrt = math.sqrt(g['g11'])
            g22_sqrt = math.sqrt(g['g22'])
            x = basis['e1'][0] * du * g11_sqrt + basis['e2'][0] * dv * g22_sqrt
            y = basis['e1'][1] * du * g11_sqrt + basis['e2'][1] * dv * g22_sqrt
            return x, y
        
        x0, y0 = toLocal(u0 - centerU, v0 - centerV)
        x1, y1 = toLocal(u1 - centerU, v1 - centerV)
        x2, y2 = toLocal(u2 - centerU, v2 - centerV)
        
        K = self.getGaussianCurvature(centerU, centerV)
        
        p0 = np.array([x0, y0, -K * (x0*x0 + y0*y0) * 0.5])
        p1 = np.array([x1, y1, -K * (x1*x1 + y1*y1) * 0.5])
        p2 = np.array([x2, y2, -K * (x2*x2 + y2*y2) * 0.5])
        
        return [p0, p2, p1] if orientation < 0 else [p0, p1, p2]

    def renderLocalMesh(self, centerU, centerV, radius, orientation):
        positions = []
        normals = []
        indices = []
        vertexCount = 0
        
        for tri in self.triangles:
            triCenter = [
                (tri['v0'][0] + tri['v1'][0] + tri['v2'][0]) / 3,
                (tri['v0'][1] + tri['v1'][1] + tri['v2'][1]) / 3
            ]
            dist = self.uvDistance(centerU, centerV, triCenter[0], triCenter[1])
            
            if dist < radius:
                p0, p1, p2 = self.projectTriangleToR3(tri, centerU, centerV, orientation)
                
                v1 = p1 - p0
                v2 = p2 - p0
                n = np.cross(v1, v2)
                norm = np.linalg.norm(n)
                if norm > 0: n = n / norm
                
                positions.extend([p0, p1, p2])
                normals.extend([n, n, n])
                indices.extend([vertexCount, vertexCount + 1, vertexCount + 2])
                vertexCount += 3
        
        return (np.array(positions, dtype=np.float32),
                np.array(normals, dtype=np.float32),
                np.array(indices, dtype=np.uint32))


# --- Funciones para crear superficies ---

def createTorus():
    surface = TopologicalSurface()
    surface.name = "Toro"
    surface.wrapU = True
    surface.wrapV = True
    surface.orientationFlipU = False
    surface.orientationFlipV = False
    # --- MODIFICADO --- Reducida la resolución para fluidez
    surface.createRegularTriangulation(25, 15, [0, 1], [0, 1])
    R, r = 3, 1
    # ... (el resto de la función es igual) ...
    def metric(u, v):
        phi = v * 2 * math.pi
        return {
            'g11': math.pow(R + r * math.cos(phi), 2) * math.pow(2 * math.pi, 2),
            'g12': 0,
            'g22': math.pow(r * 2 * math.pi, 2)
        }
    surface.metric = metric
    def curvature(u, v):
        phi = v * 2 * math.pi
        cos_phi = math.cos(phi)
        denominator = r * (R + r * cos_phi)
        return cos_phi / denominator if denominator != 0 else 0
    surface.curvature = curvature
    surface.landmarks = [
        {'u': 0, 'v': 0, 'label': "A", 'color': (1, 0, 0)},
        {'u': 0.5, 'v': 0, 'label': "B", 'color': (0, 1, 0)},
        {'u': 0, 'v': 0.5, 'label': "C", 'color': (0, 0, 1)},
        {'u': 0.5, 'v': 0.5, 'label': "D", 'color': (1, 1, 0)}
    ]
    return surface

def createMoebiusStrip():
    surface = TopologicalSurface()
    surface.name = "Banda de Möbius"
    surface.wrapU = True
    surface.wrapV = False
    surface.orientationFlipU = True
    surface.orientationFlipV = False
    # --- MODIFICADO --- Reducida la resolución para fluidez
    surface.createRegularTriangulation(30, 10, [0, 1], [-0.3, 0.3])
    surface.metric = lambda u, v: {'g11': 40, 'g12': 0, 'g22': 2.25}
    surface.curvature = lambda u, v: 0
    surface.landmarks = [
        {'u': 0.5, 'v': 0, 'label': "Centro", 'color': (0, 1, 0)} # Movido al centro
    ]
    return surface

def createKleinBottle():
    surface = TopologicalSurface()
    surface.name = "Botella de Klein"
    surface.wrapU = True
    surface.wrapV = True
    surface.orientationFlipU = True
    surface.orientationFlipV = False
    # --- MODIFICADO --- Reducida la resolución para fluidez
    surface.createRegularTriangulation(25, 15, [0, 1], [0, 1])
    surface.metric = lambda u, v: {'g11': math.pow(2 * math.pi, 2) * 4, 'g12': 0, 'g22': math.pow(2 * math.pi, 2)}
    surface.curvature = lambda u, v: 0.2 * math.sin(u * 2 * math.pi) * math.cos(v * 2 * math.pi)
    surface.landmarks = [
        {'u': 0.35, 'v': 0.25, 'label': "A", 'color': (1, 0, 0)},
        {'u': 0.65, 'v': 0.75, 'label': "B", 'color': (0, 0, 1)}
    ]
    return surface

def createProjectivePlane():
    surface = TopologicalSurface()
    surface.name = "Plano Proyectivo"
    surface.wrapU = True
    surface.wrapV = True
    surface.orientationFlipU = True
    surface.orientationFlipV = True
    # --- MODIFICADO --- Reducida la resolución para fluidez
    surface.createRegularTriangulation(20, 20, [0, 1], [0, 1])
    def metric(u, v):
        theta = v * math.pi
        sin_theta = math.sin(theta)
        if sin_theta == 0: sin_theta = 0.0001
        return {'g11': math.pow(math.pi, 2) * math.pow(sin_theta, 2), 'g12': 0, 'g22': math.pow(math.pi, 2)}
    surface.metric = metric
    surface.curvature = lambda u, v: 1
    surface.landmarks = [
        {'u': 0.25, 'v': 0.5, 'label': "E", 'color': (0, 0, 1)}
    ]
    return surface


# --- AÑADIR ESTA FUNCIÓN COMPLETA EN surfaces.py ---

def createMoebiusStrip2():
    surface = TopologicalSurface()
    surface.name = "Banda de Möbius 2 (con colina)"
    surface.wrapU = True
    surface.wrapV = False
    surface.orientationFlipU = True
    surface.orientationFlipV = False
    
    # El doble de resolución para un mundo más grande
    surface.createRegularTriangulation(30, 10, [0, 1], [-0.3, 0.3])
    
    # --- 1. MÉTRICA (DOBLE TAMAÑO) ---
    # Multiplicamos g11 (40*4=160) y g22 (2.25*4=9) por 4
    surface.metric = lambda u, v: {'g11': 160, 'g12': 0, 'g22': 9}

    # --- 2. CURVATURA (LA COLINA) ---
    def curvature(u, v):
        u_c, v_c = 0.5, 0.0  # Centro de la colina
        hill_radius = 0.15  # Radio de la colina
        hill_height = 8.0   # Curvatura máxima en el pico (puedes ajustar esto)

        # Calculamos la distancia en el mapa UV, respetando el "wrap" de U
        du = abs(u - u_c)
        du = min(du, 1 - du) # Moebius se envuelve en U
        dv = abs(v - v_c)    # No se envuelve en V

        r_sq = du*du + dv*dv
        radius_sq = hill_radius * hill_radius
        
        if r_sq >= radius_sq:
            return 0  # Estamos fuera de la colina
        
        # Fórmula de la C-infinity bump function
        # Esto es e^(1 - 1/(1 - (r/R)^2))
        r_norm_sq = r_sq / radius_sq
        return hill_height * math.exp(1.0 - 1.0 / (1.0 - r_norm_sq))
    
    surface.curvature = curvature
    
    # El landmark está en el centro de la colina
    surface.landmarks = [
        {'u': 0.5, 'v': 0, 'label': "Colina", 'color': (0, 1, 0)}
    ]
    return surface