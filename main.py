import pygame
import random
import math
from PIL import Image

def find_intersections(radius, angle_degrees):
    # Convert angle from degrees to radians
    theta = math.radians(angle_degrees % 180)

    # Calculate coordinates of the intersection points
    x1 = radius * math.cos(theta)
    y1 = radius * math.sin(theta)
    x2 = -x1
    y2 = -y1

    angle_degrees = angle_degrees % 360
    #angle_degrees < 90 or angle_degrees > 270
    return (x1, y1) if angle_degrees < 180 else (x2, y2)

def calculate_angle(x, y):
    # Calculate the angle in radians
    angle_radians = math.atan2(y, x)

    # Optionally convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

def calculate_distance(x, y):
    # Calculate the Euclidean distance from the origin to the point (x, y)
    distance = math.sqrt(x**2 + y**2)
    return distance


def point_on_unit_circle(degrees, moveBy=1.0):
    # Convert degrees to radians
    radians = math.radians(degrees)

    # Calculate coordinates on the unit circle
    x = math.cos(radians)
    y = math.sin(radians)

    return (x*moveBy, y*moveBy)


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def point_in_triangle(pt, v1, v2, v3):
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def pixels_in_triangle(v1, v2, v3):
    # Determine the bounding box of the triangle
    min_x = min(v1[0], v2[0], v3[0])
    max_x = max(v1[0], v2[0], v3[0])
    min_y = min(v1[1], v2[1], v3[1])
    max_y = max(v1[1], v2[1], v3[1])

    pixels = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if point_in_triangle((x, y), v1, v2, v3):
                pixels.append((x, y))

    return pixels

###
###
###

class Coordinate:
    def __init__(self, val=None):
        if val is None:
            val = [0, 0, 0]
        self.val = val
        self.transformed = None

    @property
    def x(self):
        return self.val[0]

    @x.setter
    def x(self, value):
        self.val[0] = value

    @property
    def y(self):
        return self.val[1]

    @y.setter
    def y(self, value):
        self.val[1] = value

    @property
    def z(self):
        return self.val[2]

    @z.setter
    def z(self, value):
        self.val[2] = value

    def __sub__(self, other):
        return Coordinate([self.x - other.x, self.y - other.y, self.z - other.z])

    def __add__(self, other):
        return Coordinate([self.x + other.x, self.y + other.y, self.z + other.z])

    def __mul__(self, other):
        return Coordinate([self.x * other, self.y * other, self.z * other])

    def __truediv__(self, other):
        return Coordinate([self.x / other, self.y / other, self.z / other])

class Point3D:
    def __init__(self):
        self.position = Coordinate()
        self.rotation = Coordinate()
        self.transformedPosition = None
        self.transformedRotation = None
        self.parent = None

class Camera(Point3D):
    def __init__(self):
        super().__init__()
        self.fov = 1

def calcRotation(rel, rotation):
    rel = rel*1

    radius = calculate_distance(rel.x, rel.y)
    angle = calculate_angle(rel.x, rel.y)
    intersection = find_intersections(radius, rotation.x + angle)
    rel.x = intersection[0]
    rel.y = intersection[1]

    radius = calculate_distance(rel.y, rel.z)
    angle = calculate_angle(rel.y, rel.z)
    intersection = find_intersections(radius, rotation.y + angle)
    rel.y = intersection[0]
    rel.z = intersection[1]

    radius = calculate_distance(rel.x, rel.z)
    angle = calculate_angle(rel.x, rel.z)
    intersection = find_intersections(radius, rotation.z + angle)
    rel.x = intersection[0]
    rel.z = intersection[1]

    return rel

class Group(Point3D):
    def __init__(self):
        super().__init__()
        self.children = []
        self.texture = None

    def add(self, obj):
        self.children.append(obj)
        obj.parent = self

    def transform(self, position=None, rotation=None):

        if position is None:
            position = self.position * 1
        else:
            position = position + self.position

        newPos = calcRotation(self.position, self.rotation+rotation)
        position += calcRotation(newPos, rotation)

        for c in range(0, len(self.children)):
            child = self.children[c]

            cPos = child
            if isinstance(child, Coordinate):
                cPos = child # if child.transformed is None else child.transformed
            else:
                cPos = child.position # if child.transformedPosition is None else child.transformedPosition

            rel = cPos * 1 #+ position

            rel += position
            if isinstance(child, Group):
                rel = child.transform(rel, rotation)

            #rel -= position
            rel = calcRotation(rel, self.rotation + rotation)
            #rel += position
            #rel = calcRotation(rel, rotation)

            if isinstance(child, Coordinate):
                child.transformed = rel
            else:
                child.transformedPosition = rel

        return position

    def reset(self):
        for child in self.children:
            if isinstance(child, Group):
                child.reset()

            if isinstance(child, Coordinate):
                child.transformed = None
            else:
                child.transformedPosition = None
                child.transformedRotation = None

    def list(self):
        res = []
        for child in self.children:
            if isinstance(child, Group) and not isinstance(child, Triangle):
                res += child.list()
            else:
                res.append(child)
        return res

class Triangle(Group):
    def __init__(self):
        super().__init__()
        self.vertices = []
        self.children = self.vertices
        for i in range(0,3):
            self.vertices.append(Coordinate())

class Mesh(Group):
    def __init__(self):
        super().__init__()
        self.texture = None

def drawCircle(pygame, x, y, radius):
    circle_color = (255, 0, 0)  # Red

    # Draw the circle
    pygame.draw.circle(screen, circle_color, (x,y), radius)

class Camera(Group):
    def __init__(self):
        super().__init__()
        self.fov = 10

    def render(self, scene, pygame, screen):
        scene.reset()
        scene.transform(self.position, self.rotation)

        width = screen.get_width()
        height = screen.get_height()

        avgFov = ((width+height)/2)
        fov = self.fov

        midWidth = width/2
        midHeight = height/2

        def posToScreen(pos):
            diffZ = pos.z
            if diffZ != 0:
                diffZ = fov / diffZ
                diffZ /= fov
                pos.x *= diffZ
                pos.y *= diffZ
                pos.z *= diffZ

                x = int((pos.x * avgFov) + midWidth)
                y = int((pos.y * avgFov) + midHeight)
                return Coordinate([x, y, pos.z])

            return Coordinate()

        list = scene.list()
        drawTexture = []
        for obj in list:
            if isinstance(obj, Triangle):
                if obj.transformedPosition.z < 0:
                    vertices = []
                    for vertex in obj.vertices:
                        pos = posToScreen(vertex.transformed)
                        if pos.x > 0 and pos.x < width:
                            if pos.y > 0 and pos.y < height:
                                vertices.append(pos)

                    for i in range(0, len(vertices)):
                        next = (i+1)%len(vertices)
                        if vertices[i].z > 0 or vertices[next].z > 0:
                            pygame.draw.line(screen, (0,255,0), (vertices[i].x, vertices[i].y), (vertices[next].x, vertices[next].y), 2)

                    if len(vertices)==3 and obj.parent is not None and obj.parent.texture is not None:
                        if obj.parent not in drawTexture:
                            drawTexture.append(obj.parent)

                        minX = 999999
                        maxX = -1
                        minY = 999999
                        maxY = -1

                        for vertex in vertices:
                            if minX > vertex.x:
                                minX = vertex.x
                            if maxX < vertex.x:
                                maxX = vertex.x
                            if minY > vertex.y:
                                minY = vertex.y
                            if maxY < vertex.y:
                                maxY= vertex.y

                        obj.drawRange = [[minX, maxX-minX], [minY, maxY-minY]]
                        obj.textureArea = pixels_in_triangle(vertices[0].val, vertices[1].val, vertices[2].val)
            else:
                if obj.transformedPosition.z < 0:
                    pos = posToScreen(obj.transformedPosition)
                    if pos.x > 0 and pos.x < width:
                        if pos.y > 0 and pos.y < height:
                            drawCircle(pygame, pos.x, pos.y, pos.z)

        for mesh in drawTexture:
            for child in mesh.children:
                if child.drawRange[0][1] == 0 or child.drawRange[1][1] == 0:
                    continue

                for pixel in child.textureArea:
                    x = (pixel[0] - child.drawRange[0][0]) / child.drawRange[0][1]
                    y = (pixel[1] - child.drawRange[1][0]) / child.drawRange[1][1]

                    x = int(x*(mesh.texture.width-1))
                    y = int(y*(mesh.texture.height-1))

                    screen.set_at(pixel, mesh.texture.getpixel((x, y)))

class Scene(Group):
    def __init__(self):
        super().__init__()

scene = Scene()

point = Point3D()
point.position = Coordinate([0,1,1])

triangle = Triangle()
triangle.vertices[0] = Coordinate([0,1,0])
triangle.vertices[1] = Coordinate([-1,0,0])
triangle.vertices[2] = Coordinate([1,0,0])

triangle2 = Triangle()
triangle2.vertices[0] = Coordinate([0,1,0])
triangle2.vertices[1] = Coordinate([-1,0,0])
triangle2.vertices[2] = Coordinate([1,0,0])

mesh = Mesh()
mesh.add(triangle2)
mesh.position.z = -1
mesh.texture = Image.open('rainbow.jpeg')

scene.add(point)
scene.add(triangle)
scene.add(mesh)

camera = Camera()
camera.position.z = -10

# Initialize Pygame
pygame.init()

# Window size
width, height = 800, 600
screen = pygame.display.set_mode((width, height))

# Set the title of the window
pygame.display.set_caption('Pixel Drawing')

# Main loop flag
running = True

# Main loop
keyPressing = None
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:  # Check for key presses
            keyPressing = event.key
        if event.type == pygame.KEYUP:  # Check for key presses
            keyPressing = None

    if keyPressing is not None:
        moveBy = 0.01
        if keyPressing == pygame.K_UP:
            move = point_on_unit_circle(camera.rotation.z, moveBy)
            camera.position.z += move[0]
            camera.position.x += move[1]
        elif keyPressing == pygame.K_DOWN:
            move = point_on_unit_circle(camera.rotation.z, moveBy)
            camera.position.z -= move[0]
            camera.position.x -= move[1]
        elif keyPressing == pygame.K_LEFT:
            camera.rotation.z -= moveBy * 10
        elif keyPressing == pygame.K_RIGHT:
            camera.rotation.z += moveBy * 10

    screen.fill((0,0,0))
    camera.render(scene, pygame, screen)
    pygame.display.flip()  # Update the full display Surface to the screen

# Quit Pygame
pygame.quit()
