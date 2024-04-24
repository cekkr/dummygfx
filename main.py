from concurrent.futures import ThreadPoolExecutor

import numpy
import pygame
import random
import math
from PIL import Image
import concurrent.futures
import numpy as np
import time
#from OpenGL.GL import *
#from OpenGL.GLU import *
#from pygame.locals import *
import pyopencl as cl
import asyncio
import threading
from threading import Thread, Lock

# Set up OpenCL context and queue
platform = cl.get_platforms()[0]
devices = platform.get_devices()
device = devices[-1]
context = cl.Context([device])
queue = cl.CommandQueue(context)

kernel_code = """
__kernel void rotatePoints(__global float *coords, __global float *rotations, __global float *results, const int num_points) {
    int i = get_global_id(0);
    if (i >= num_points) return;

    // Each point has x, y, z values, so index should be 3 times the point index
    int idx = i * 3;
    float x = coords[idx];
    float y = coords[idx + 1];
    float z = coords[idx + 2];

    // Rotations are passed as x, y, z for each point
    float rotation_x = rotations[idx];
    float rotation_y = rotations[idx + 1];
    float rotation_z = rotations[idx + 2];

    float radius, angle, sin_theta, cos_theta;

    if (rotation_x != 0) {
        radius = sqrt(x * x + y * y);
        angle = atan2(y, x) + radians(rotation_x);
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        x = radius * cos_theta;
        y = radius * sin_theta;
    }
    
    if (rotation_y != 0) {
        radius = sqrt(y * y + z * z);
        angle = atan2(z, y) + radians(rotation_y);
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        y = radius * cos_theta;
        z = radius * sin_theta;
    }
            
    if (rotation_z != 0) {
        radius = sqrt(x * x + z * z);
        angle = atan2(x, z) + radians(rotation_z*-1);
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        z = radius * cos_theta;
        x = radius * sin_theta;
    }

    // Save results
    results[idx] = x;
    results[idx + 1] = y;
    results[idx + 2] = z;
}
"""

program = cl.Program(context, kernel_code).build()

'''
def cl_rotatePoints(reqs):
    coords = [point[0] for point in reqs]
    rotations = [point[1] for point in reqs]

    # Create buffers for the data
    coords = np.array(coords).astype(np.float32)
    rotations = np.array(rotations).astype(np.float32)
    results = np.empty_like(coords)

    coords_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=coords)
    rotations_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=rotations)
    results_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, coords.nbytes)

    # Run the kernel
    num_points = len(reqs)
    program.rotatePoints(queue, (num_points,), None, coords_buf, rotations_buf, results_buf, np.int32(num_points))

    # Read back the results
    cl.enqueue_copy(queue, results, results_buf)

    resCoords = []
    for res in results:
        resCoords.append(Coordinate(res))
    return resCoords
'''

class Event_ts(asyncio.Event):
    def set(self):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(super().set)
        else:
            return super().set()

    def clear(self):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(super().clear)
        else:
            return super().clear()

    def wait(self):
        return super().wait()
        if self._loop is not None:
            return self._loop.call_soon_threadsafe(super().wait)
        else:
            return super().wait()

# Initialize OpenCL context and queue
executor = ThreadPoolExecutor()
request_buffer = []
buffer_lock = asyncio.Lock()
batch_size = 10  # Set minimum number of requests before processing
batch_processed_event = Event_ts()
#batch_processed_event.clear()

def synchronous_cl_rotate_points(reqs):
    # Ensure that the context and queue are properly initialized
    global context, queue, program

    # Prepare data arrays
    coords = np.array([point[0] for point in reqs], dtype=np.float32)
    rotations = np.array([point[1] for point in reqs], dtype=np.float32)
    results = np.empty_like(coords)  # This will store the output

    # Create OpenCL buffers
    coords_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=coords)
    rotations_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=rotations)
    results_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=results.nbytes)

    # Execute the kernel
    num_points = len(reqs)
    program.rotatePoints(queue, (num_points,), None, coords_buf, rotations_buf, results_buf, np.int32(num_points))

    # Read back the results
    cl.enqueue_copy(queue, results, results_buf)
    queue.finish()  # Ensure all queued operations are completed

    # Convert the results into Coordinate objects
    resCoords = [Coordinate(res) for res in results]
    return resCoords

async def cl_rotatePoints(new_reqs):
    global batch_processed_event

    #async with buffer_lock:
    startFrom = len(request_buffer)
    request_buffer.extend(new_reqs)
    current_buffer_size = len(request_buffer)

    '''
    if current_buffer_size >= batch_size:
        await process_batch()
    else:
        #batch_processed_event.clear()
        pass
    '''

    #await monitor_cl_rotate_points()

    await batch_processed_event.wait()  # Wait until the batch is processed
    res = results[startFrom:current_buffer_size]  # Return processed results for the original request count

    return res

curBatchCycle = 0
results = []
async def process_batch():
    global results
    global request_buffer
    global curBatchCycle

    request = request_buffer
    request_buffer = []

    #loop = asyncio.get_running_loop()
    #results = await loop.run_in_executor(executor, synchronous_cl_rotate_points, request)
    results = synchronous_cl_rotate_points(request)

    batch_processed_event.set()  # Notify waiting tasks that the batch has been processed
    batch_processed_event.clear()

    curBatchCycle += 1

last_time_modified = None
last_size_checked = 0
time_threshold = 0.001  # 10 milliseconds
batchCycle = -1

async def monitor_cl_rotate_points():
    global last_time_modified, last_size_checked, time_threshold, batchCycle

    current_time = time.time()

    current_size = len(request_buffer)
    # Check if the buffer has been modified or if it's the first time
    if current_size != last_size_checked:
        last_size_checked = current_size
        last_time_modified = current_time
    # Check if the time threshold has been reached with no size change

    if (current_size > 0 and (current_time - last_time_modified) >= time_threshold) :# and batchCycle < curBatchCycle:
        print("10 ms have passed with no change in buffer size. Processing batch... ", current_size)
        batchCycle = curBatchCycle
        await process_batch()
        last_size_checked = 0  # Reset after processing
        last_time_modified = current_time

async def monitor_cl_rotate_points_loop():
    while True:
        await asyncio.sleep(time_threshold)  # Sleep for 1 millisecond to prevent high CPU usage
        await monitor_cl_rotate_points()

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
    return ((p1[0] - p3[0]) * (p2[1] - p3[1])) - ((p2[0] - p3[0]) * (p1[1] - p3[1]))


def point_in_triangle(pt, v1, v2, v3):
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def pixels_in_triangle(v1, v2, v3):
    v1 = [int(v1[0]), int(v1[1])]
    v2 = [int(v2[0]), int(v2[1])]
    v3 = [int(v3[0]), int(v3[1])]

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


def find_line_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    # Calculate the slope
    if x2 - x1 == 0:
        # Avoid division by zero; handle vertical lines separately if necessary
        return float('inf'), y1

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    return m, b


def x_from_y(m, b, y):
    if m == 0:
        return float('inf')

    if m == float('inf'):
        return b

    x = (y - b) / m
    return x

def list_pixels_in_triangle(v1, v2, v3, size):
    v1 = [int(v1[0]), int(v1[1])]
    v2 = [int(v2[0]), int(v2[1])]
    v3 = [int(v3[0]), int(v3[1])]

    # Determine the bounding box of the triangle
    min_x = min(v1[0], v2[0], v3[0])
    max_x = max(v1[0], v2[0], v3[0])
    min_y = min(v1[1], v2[1], v3[1])
    max_y = max(v1[1], v2[1], v3[1])

    vv = [v1,v2,v3]
    m1, b1 = find_line_equation(v1, v2)
    m2, b2 = find_line_equation(v2, v3)
    m3, b3 = find_line_equation(v3, v1)

    range1 = [v1[1], v2[1]]
    range2 = [v2[1], v3[1]]
    range3 = [v3[1], v1[1]]

    range1.sort()
    range2.sort()
    range3.sort()

    def y_in_range(range, y):
        return y >= range[0] and y <= range[1]

    pixels = []
    for y in range(min_y, max_y + 1):
        xx = []
        if y_in_range(range1, y):
            xx.append(x_from_y(m1, b1, y))
        if y_in_range(range2, y):
            xx.append(x_from_y(m2, b2, y))
        if y_in_range(range3, y):
            xx.append(x_from_y(m3, b3, y))

        x = 0
        while x < len(xx):
            if xx[x] == float('inf'):
                del xx[x]
                x -= 1
            x += 1

        xx.sort()

        for x in range(int(xx[0]), int(xx[1])):
            if 0 <= x < size[0] and 0 <= y <= size[1]:
                pixels.append((x,y))

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

    def isZero(self):
        return self.x == 0 and self.y == 0 and self.z == 0

class Point3D:
    def __init__(self):
        self.position = Coordinate()
        self.rotation = Coordinate()
        self.transformed = None
        self.transformedRotation = None
        self.parent = None

class Camera(Point3D):
    def __init__(self):
        super().__init__()
        self.fov = 1

def calcRotation(rel, rotation):
    rel = rel*1

    if rotation.x != 0:
        radius = calculate_distance(rel.x, rel.y)
        angle = calculate_angle(rel.x, rel.y)
        intersection = find_intersections(radius, rotation.x + angle)
        rel.x = intersection[0]
        rel.y = intersection[1]

    if rotation.y != 0:
        radius = calculate_distance(rel.y, rel.z)
        angle = calculate_angle(rel.y, rel.z)
        intersection = find_intersections(radius, rotation.y + angle)
        rel.y = intersection[0]
        rel.z = intersection[1]

    if rotation.z != 0:
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

    async def transform(self, position=None, rotation=None):
        if position is None:
            position = self.position * 1
        else:
            position = position + self.position

        totRotation = self.rotation + rotation

        #if self.transformed is not None and self.lastRotation.val == totRotation.val:
        #    return self.transformed

        if not position.isZero():
            newPos = self.position
            if not totRotation.isZero() and not newPos.isZero():
                newPos = await cl_rotatePoints([[newPos.val, totRotation.val]])
                newPos = newPos[0]

            if not rotation.isZero() and not newPos.isZero():
                newPos = await cl_rotatePoints([[newPos.val, rotation.val]])
                newPos = newPos[0]

            position += newPos

        async def process_child(child):
            if isinstance(child, Coordinate):
                cPos = child  # if child.transformed is None else child.transformed
            else:
                cPos = child.position  # if child.transformed is None else child.transformed

            rel = cPos * 1
            rel += position
            if isinstance(child, Group):
                rel = await child.transform(rel, rotation)

            if rel.isZero() or totRotation.isZero():
                return rel

            #if child.transformed is not None and totRotation.val == child.lastRotation.val:
            #    return child.transformed

            rel = await cl_rotatePoints([[rel.val, totRotation.val]])
            return rel[0]

        # Use ThreadPoolExecutor to process children in parallel
        tasks = [process_child(child) for child in self.children]
        res = await asyncio.gather(*tasks)
        #res = await cl_rotatePoints(reqs)

        for i, child in enumerate(self.children):
            rel = res[i]

            child.transformed = rel
            child.lastRotation = totRotation

        self.transformed = position
        self.lastRotation = totRotation

        return position

        '''
        newPos = calcRotation(self.position, self.rotation+rotation)
        position += calcRotation(newPos, rotation)

        for c in range(0, len(self.children)):
            child = self.children[c]

            cPos = child
            if isinstance(child, Coordinate):
                cPos = child # if child.transformed is None else child.transformed
            else:
                cPos = child.position # if child.transformed is None else child.transformed

            rel = cPos * 1

            rel += position
            if isinstance(child, Group):
                rel = child.transform(rel, rotation)

            rel = calcRotation(rel, self.rotation + rotation)

            if isinstance(child, Coordinate):
                child.transformed = rel
            else:
                child.transformed = rel

        return position
        '''

    def reset(self):
        for child in self.children:
            if isinstance(child, Group):
                child.reset()

            child.transformed = None
            #child.transformedRotation = None

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

    def setTexture(self, image):
        if image.mode != 'RGBA':
            image = image.convert('RGBA')  # Convert to RGBA if not already in this mode

        self.image = image

        # Convert the PIL image to a string buffer and then to a Pygame surface
        raw_str = image.tobytes("raw", 'RGBA')
        image_size = image.size

        # Create a Pygame Surface
        pygame_surface = pygame.image.fromstring(raw_str, image_size, 'RGBA')

        self.texture = pygame_surface
        self.texture_array = pygame.surfarray.array3d(self.texture)

    def loadModelTxt(self, path):
        with open(path, "r") as file:
            # Read the entire contents of the file into a string
            content = file.read()

        triangles = content.split('\n')
        for t in triangles:
            if t == '':
                break

            triangle = Triangle()
            vertices = t.split(' ')
            for v in range(0, 3):
                vertex = vertices[v].split(',')
                triangle.vertices[v] = Coordinate([float(vertex[0]), float(vertex[1]), float(vertex[2])])

            self.add(triangle)

def drawCircle(pygame, x, y, radius, screen):
    circle_color = (255, 0, 0)  # Red

    # Draw the circle
    pygame.draw.circle(screen, circle_color, (x,y), radius)

def apply_texture(child, mesh, screen, screen_array):
    texture_array = mesh.texture_array

    width = mesh.image.width
    height = mesh.image.height

    screenWidth = screen.get_width()
    screenHeight = screen.get_height()

    vv = [[child.screenVertices[0].val[0],child.screenVertices[0].val[1]],[child.screenVertices[1].val[0],child.screenVertices[1].val[1]],[child.screenVertices[2].val[0],child.screenVertices[2].val[1]]]

    # Determine the bounding box of the triangle
    min_x = min(vv[0][0], vv[1][0], vv[2][0])
    max_x = max(vv[0][0], vv[1][0], vv[2][0])
    min_y = min(vv[0][1], vv[1][1], vv[2][1])
    max_y = max(vv[0][1], vv[1][1], vv[2][1])

    m1, b1 = find_line_equation(vv[0], vv[1])
    m2, b2 = find_line_equation(vv[1], vv[2])
    m3, b3 = find_line_equation(vv[2], vv[0])

    range1 = [vv[0][1], vv[1][1]]
    range2 = [vv[1][1], vv[2][1]]
    range3 = [vv[2][1], vv[0][1]]

    range1.sort()
    range2.sort()
    range3.sort()

    def y_in_range(range, y):
        return y >= range[0] and y <= range[1]

    pixels = []
    for y in range(min_y, max_y + 1):
        xx = []
        if y_in_range(range1, y):
            xx.append(x_from_y(m1, b1, y))
        if y_in_range(range2, y):
            xx.append(x_from_y(m2, b2, y))
        if y_in_range(range3, y):
            xx.append(x_from_y(m3, b3, y))

        x = 0
        while x < len(xx):
            if xx[x] == float('inf'):
                del xx[x]
                x -= 1
            x += 1

        xx.sort()

        yy = (y - child.drawRange[1][0]) / child.drawRange[1][1]
        yy = int(yy*(height-1))

        diff = xx[1] - xx[0]
        if diff > 0:
            x1 = ((xx[0] - child.drawRange[0][0]) / child.drawRange[0][1])*(width-1)
            x2 = ((xx[1] - child.drawRange[0][0]) / child.drawRange[0][1])*(width-1)
            xInc = (x2-x1) / diff

            for i in range(0, int(diff)):
                x = i+int(xx[0])

                if not (0 <= x < screenWidth and 0 <= y < screenHeight):
                    continue

                screen_array[x, y] = texture_array[int(x1), yy]
                x1 += xInc

    new_surface = pygame.surfarray.make_surface(screen_array)
    screen.blit(new_surface, (0, 0))

'''
def apply_texture_chunk(texture_array, width, height, drawRange, screen_array, pixels):
    for pixel in pixels:
        x = (pixel[0] - drawRange[0][0]) / drawRange[0][1]
        y = (pixel[1] - drawRange[1][0]) / drawRange[1][1]

        x = int(x * width)
        y = int(y * height)

        # Directly set pixels in the screen's array
        if 0 <= x < width and 0 <= y < height:
            screen_array[pixel[0], pixel[1]] = texture_array[x, y]

def apply_texture(child, mesh, screen, screen_array):
    texture_array = mesh.texture_array
    width = mesh.image.width
    height = mesh.image.height
    textureArea = child.textureArea
    num_threads = 2  # You can adjust this number based on your application needs

    # Splitting textureArea into chunks
    chunk_size = len(textureArea) // num_threads
    chunks = [textureArea[i * chunk_size:(i + 1) * chunk_size] for i in range(num_threads)]
    chunks.append(textureArea[(num_threads - 1) * chunk_size:])  # Adding any leftover pixels

    # Use ThreadPoolExecutor to process each chunk in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Map the apply_texture_chunk to each chunk
        futures = [executor.submit(apply_texture_chunk, texture_array, width, height, child.drawRange, screen_array, chunk) for chunk in chunks]

        # Optionally, ensure all futures are done (they will be upon exiting the context manager)
        concurrent.futures.wait(futures)

    # Creating a new surface and blitting it
    new_surface = pygame.surfarray.make_surface(screen_array)
    screen.blit(new_surface, (0, 0))
    pygame.display.flip()
'''

class Camera(Group):
    def __init__(self):
        super().__init__()
        self.fov = 10

    async def render(self, scene, pygame, screen):
        #scene.reset()
        await scene.transform(self.position, self.rotation)

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
                if obj.transformed.z < 0:
                    vertices = []
                    somethingInside = False
                    for vertex in obj.vertices:
                        pos = posToScreen(vertex.transformed)
                        vertices.append(pos)
                        if pos.x > 0 and pos.x < width:
                            if pos.y > 0 and pos.y < height:
                                somethingInside = True

                    if not somethingInside:
                        continue

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
                        obj.screenVertices = vertices
                        #obj.textureArea = list_pixels_in_triangle(vertices[0].val, vertices[1].val, vertices[2].val, [width, height])
            else:
                if obj.transformed.z < 0:
                    pos = posToScreen(obj.transformed)
                    if pos.x > 0 and pos.x < width:
                        if pos.y > 0 and pos.y < height:
                            drawCircle(pygame, pos.x, pos.y, int(pos.z), screen)

        screen_array = pygame.surfarray.array3d(screen)

        for mesh in drawTexture:
            for child in mesh.children:
                if child.drawRange[0][1] == 0 or child.drawRange[1][1] == 0:
                    continue

                apply_texture(child, mesh, screen, screen_array)

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

async def main():
    #asyncio.create_task(monitor_cl_rotate_points())

    scene = Scene()

    point = Point3D()
    point.position = Coordinate([0,1,1])

    if False:
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
        mesh.setTexture(Image.open('rainbow.jpeg'))

        scene.add(point)
        scene.add(triangle)
        scene.add(mesh)
    else:
        mesh = Mesh()
        mesh.loadModelTxt('complexModel.txt')
        scene.add(mesh)

    camera = Camera()
    camera.position.z = -10

    # Initialize Pygame
    pygame.init()

    # Window size
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height), DOUBLEBUF)

    # Set the title of the window
    pygame.display.set_caption('Pixel Drawing')

    # Main loop flag
    running = True

    # Main loop
    keyPressing = None
    fps = []
    avgFps = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:  # Check for key presses
                keyPressing = event.key
            if event.type == pygame.KEYUP:  # Check for key presses
                keyPressing = None

        if keyPressing is not None:
            moveBy = 0.1 * (120/avgFps)
            if keyPressing == pygame.K_UP:
                move = point_on_unit_circle(camera.rotation.z, moveBy)
                camera.position.z += move[0]
                camera.position.x += move[1]
            elif keyPressing == pygame.K_DOWN:
                move = point_on_unit_circle(camera.rotation.z, moveBy)
                camera.position.z -= move[0]
                camera.position.x -= move[1]
            elif keyPressing == pygame.K_LEFT:
                camera.rotation.z -= moveBy * 5
            elif keyPressing == pygame.K_RIGHT:
                camera.rotation.z += moveBy * 5

        screen.fill((0,0,0))
        await camera.render(scene, pygame, screen)

        font = pygame.font.SysFont('Arial', 12)
        text_surface = font.render(str(avgFps), False, (255, 255, 255))
        screen.blit(text_surface, (5, 5))

        pygame.display.flip()  # Update the full display Surface to the screen

        curTime = time.time()
        fps.append(curTime)
        while len(fps) > 0 and fps[0] < curTime-0.25:
            del fps[0]
        avgFps = len(fps)*4

    # Quit Pygame
    pygame.quit()

async def run():
    task1 = asyncio.create_task(monitor_cl_rotate_points_loop())
    task2 = asyncio.create_task(main())

    # You can wait for the specific tasks if necessary, or just run indefinitely
    await asyncio.gather(task1, task2)

asyncio.run(run())