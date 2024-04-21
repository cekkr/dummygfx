import pygame
import random
import math

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

class Point3D:
    def __init__(self):
        self.position = Coordinate()
        self.rotation = Coordinate()
        self.transformedPosition = None
        self.transformedRotation = None

class Camera(Point3D):
    def __init__(self):
        super().__init__()
        self.fov = 1

class Group(Point3D):
    def __init__(self):
        super().__init__()
        self.children = []

    def add(self, obj):
        self.children.append(obj)

    def transform(self, position=None, rotation=None):
        for c in range(0, len(self.children)):
            child = self.children[c]

            if isinstance(child, Group):
                child.transform()

            cPos = child
            if isinstance(child, Coordinate):
                child = child if child.transformed is None else child.transformed
            else:
                cPos = (child.transformedPosition if child.transformedPosition is not None else child.position)

            rel = cPos - self.position

            radius = calculate_distance(rel.x, rel.y)
            angle = calculate_angle(rel.x, rel.y)
            intersection = find_intersections(radius, self.rotation.x+angle)
            rel.x = intersection[0]
            rel.y = intersection[1]

            radius = calculate_distance(rel.y, rel.z)
            angle = calculate_angle(rel.y, rel.z)
            intersection = find_intersections(radius, self.rotation.y + angle)
            rel.y = intersection[0]
            rel.z = intersection[1]

            radius = calculate_distance(rel.x, rel.z)
            angle = calculate_angle(rel.x, rel.z)
            intersection = find_intersections(radius, self.rotation.z + angle)
            rel.x = intersection[0]
            rel.z = intersection[1]

            rel += self.position
            if position is not None:
                rel += position

            if isinstance(child, Coordinate):
                child.transformed = rel
            else:
                child.transformedPosition = rel

            if isinstance(child, Group):
                child.transform(rel)

            '''
            if isinstance(child, Coordinate):
                child.transformed += self.position
            else:
                child.transformedPosition += self.position
            '''

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
        for obj in list:
            if isinstance(obj, Triangle):
                vertices = []
                for vertex in obj.vertices:
                    pos = posToScreen(vertex.transformed)
                    if pos.x > 0 and pos.x < width:
                        if pos.y > 0 and pos.y < height:
                            vertices.append(pos)

                for i in range(0, len(vertices)):
                    next = (i+1)%len(vertices)
                    pygame.draw.line(screen, (0,255,0), (vertices[i].x, vertices[i].y), (vertices[next].x, vertices[next].y), 2)
            else:
                pos = posToScreen(obj.transformedPosition)
                if pos.z > 0:
                    if pos.x > 0 and pos.x < width:
                        if pos.y > 0 and pos.y < height:
                            drawCircle(pygame, pos.x, pos.y, pos.z)

class Scene(Group):
    def __init__(self):
        super().__init__()

point = Point3D()
point.position = Coordinate([0,1,1])

triangle = Triangle()
triangle.vertices[0] = Coordinate([0,1,0])
triangle.vertices[1] = Coordinate([-1,0,0])
triangle.vertices[2] = Coordinate([1,0,0])

scene = Scene()
scene.add(point)
scene.add(triangle)

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
        moveBy = 0.1
        if keyPressing == pygame.K_UP:
            move = point_on_unit_circle(camera.rotation.z, moveBy)
            camera.position.z += move[0]
            camera.position.x += move[1]
        elif keyPressing == pygame.K_DOWN:
            move = point_on_unit_circle(camera.rotation.z, moveBy)
            camera.position.z -= move[0]
            camera.position.x -= move[1]
        elif keyPressing == pygame.K_LEFT:
            triangle.rotation.z -= moveBy
        elif keyPressing == pygame.K_RIGHT:
            triangle.rotation.z += moveBy

    screen.fill((0,0,0))
    camera.render(scene, pygame, screen)
    pygame.display.flip()  # Update the full display Surface to the screen

# Quit Pygame
pygame.quit()
