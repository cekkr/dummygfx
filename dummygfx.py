from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
from multiprocessing import Pool
import os
import functools

os.environ["PYOPENCL_CTX"] = "0"

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

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;

    if (rotation_x != 0) {
        radius = sqrt(xx + yy);
        angle = atan2(y, x) + radians(rotation_x);
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        x = radius * cos_theta;
        y = radius * sin_theta;
    }
    
    if (rotation_y != 0) {
        radius = sqrt(yy + zz);
        angle = atan2(z, y) + radians(rotation_y);
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        y = radius * cos_theta;
        z = radius * sin_theta;
    }
            
    if (rotation_z != 0) {
        radius = sqrt(xx + zz);
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

###
###
###


kernel_code = """
void rotatePoints(float x, float y, float z, float rotation_x, float rotation_y, float rotation_z, float *results) {
    float radius, angle, sin_theta, cos_theta;

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;

    if (rotation_x != 0) {
        radius = sqrt(xx + yy);
        angle = atan2(y, x) + radians(rotation_x);
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        x = radius * cos_theta;
        y = radius * sin_theta;
    }

    if (rotation_y != 0) {
        radius = sqrt(yy + zz);
        angle = atan2(z, y) + radians(rotation_y);
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        y = radius * cos_theta;
        z = radius * sin_theta;
    }

    if (rotation_z != 0) {
        radius = sqrt(xx + zz);
        angle = atan2(x, z) + radians(rotation_z*-1);
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        z = radius * cos_theta;
        x = radius * sin_theta;
    }

    // Save results
    results[0] = x;
    results[1] = y;
    results[2] = z;
}

// [position, self.rotation, (self.rotation+rotation), parent]
__kernel void calculateCommands(__global float *mainCoords, __global float *requests, __global float *results, __global int *parents, const int num_points, const int level) {
    int i = get_global_id(0);
    if (i >= num_points) return;  
    
    int idx = i*6; 
    
    int parent = parents[(i*2)];
    int thisLevel = parents[(i*2)+1];
    
    //printf("%d\\n", parent);
    
    if(thisLevel < level) return; 
    
    for(int l=0; l<level; l++)
        parent = parents[(parent*2)];

    // Each point has x, y, z values, so index should be 3 times the point index    
    float pos_x = requests[idx];
    float pos_y = requests[idx+1];
    float pos_z = requests[idx+2];
    
    float totPos_x = pos_x;
    float totPos_y = pos_y;
    float totPos_z = pos_z;
    
    float rot_x = requests[idx+3];
    float rot_y = requests[idx+4];
    float rot_z = requests[idx+5];
    
    float totRot_x = rot_x;
    float totRot_y = rot_y;
    float totRot_z = rot_z;
    
    //printf("%d\\n", level);
    
    if(parent == -1){
        totPos_x += mainCoords[0];
        totPos_y += mainCoords[1];
        totPos_z += mainCoords[2];
        
        totRot_x += mainCoords[3];
        totRot_y += mainCoords[4];
        totRot_z += mainCoords[5];
        
        rot_x = totRot_x;
        rot_y = totRot_y;
        rot_z = totRot_z;
    }
    else {
        idx = parent * 6; 
        
        totPos_x += results[idx];
        totPos_y += results[idx+1];
        totPos_z += results[idx+2];    
        
        //if(parent == 3) printf("%f %d\\n", totPos_z, level);      
        
        totRot_x = requests[idx+3];
        totRot_y = requests[idx+4];
        totRot_z = requests[idx+5];         
    }        
        
    if(level > 0){
        pos_x = totPos_x;
        pos_y = totPos_y;
        pos_z = totPos_z;
    
        rot_x = totRot_x;
        rot_y = totRot_y;
        rot_z = totRot_z;     
        
        //if(parent < 3) printf("%f \\n", rot_z);       
    }
    
    float res[3];
    //rotatePoints(totPos_x, totPos_y, totPos_z, rot_x, rot_y, rot_z, res);
    rotatePoints(pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, res);
    //rotatePoints(res[0], res[1], res[2], rot_x, rot_y, rot_z, res);    
    //rotatePoints(res[0], res[1], res[2], rot_x, rot_y, rot_z, res);
    
    /*res[0] += totPos_x;
    res[1] += totPos_y;
    res[2] += totPos_z;*/
    
    //if(parent == 3) printf("%f %f\\n", rot_z, pos_z); 
    
    /*totPos_x += res[0];
    totPos_y += res[1];
    totPos_z += res[2];*/
    
    /*res[0] += totPos_x*2;
    res[1] += totPos_y*2;
    res[2] += totPos_z*2;*/
    
    //rotatePoints(res[0], res[1], res[2], totRot_x, totRot_y, totRot_z, res);
    
    idx = i*6;
    results[idx] = res[0];
    results[idx+1] = res[1];
    results[idx+2] = res[2];
    //printf("%f\\n", res[2]);
    
    //if(parent >= 3) printf("%f \\n", res[2]);
    
    results[idx+3] = rot_x;
    results[idx+4] = rot_y;
    results[idx+5] = rot_z;
    
    for(int i=0; i<3; i++) requests[idx+i] = results[idx+i];
}
"""

#kernel_code = kernel_code.replace('float', 'double')

programCommands = cl.Program(context, kernel_code).build()

def synchronous_cl_commands(cmds, position, rotation):
    # Ensure that the context and queue are properly initialized
    global context, queue, program

    mainCoords = []
    mainCoords.extend(position.val)
    mainCoords.extend(rotation.val)

    maxLevel = 0

    requests = []
    parents = []
    cmds = cmds[::-1]
    for cmd in cmds:
        requests.append(cmd[0].val)
        requests.append(cmd[1].val)

        parent = len(cmds)-(cmd[2]+1) if cmd[2] >= 0 else -1
        parents.append([parent, cmd[3]])

        if maxLevel < cmd[3]:
            maxLevel = cmd[3]

    # Prepare data arrays
    mainCoords = np.array(mainCoords, dtype=np.float32)
    requests = np.array(requests, dtype=np.float32)
    parents = np.array(parents, dtype=np.int32)
    results = np.full((len(cmds), 6), 0, dtype=np.float32)  # This will store the output

    # Create OpenCL buffers
    mainCoords_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mainCoords)
    requests_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=requests)
    parents_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=parents)
    results_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=results)

    # Execute the kernel
    num_points = len(cmds)

    # Read back the results
    if False:
        for level in range(0, maxLevel + 1):
            level = maxLevel - level
            kernel = cl.Kernel(programCommands, "calculateCommands")
            kernel.set_args(mainCoords_buf, requests_buf, parents_buf, np.int32(num_points), np.int32(level))
            global_work_size = (num_points,)
            local_work_size = None  # or some value if needed
            cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size)
            queue.finish()
    else:
        if True:
            for level in range(0, maxLevel+1):
                #level = maxLevel - level
                #accResults_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=results)
                programCommands.calculateCommands(queue, (num_points,), None, mainCoords_buf, requests_buf, results_buf, parents_buf, np.int32(num_points), np.int32(level))
                #cl.enqueue_copy(queue, results, results_buf)
                queue.finish()

        if False:
            for level in range(0, maxLevel+1):
                level = maxLevel - level
                programCommands.calculateCommands(queue, (num_points,), None, mainCoords_buf, requests_buf, results_buf, parents_buf, np.int32(num_points), np.int32(level))
                queue.finish()

    cl.enqueue_copy(queue, results, results_buf)
    queue.finish()  # Ensure all queued operations are completed

    #print(results)

    # Convert the results into Coordinate objects
    resCoords = []
    for i in range(0, len(results)):
        ii = len(results)-(i+1)
        resCoords.append(Coordinate([results[ii][0], results[ii][1], results[ii][2]]))

    return resCoords

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
#executor = ThreadPoolExecutor()
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
    resCoords = [Coordinate(list(res)) for res in results]
    return resCoords

async def cl_rotatePoints(new_reqs):
    global batch_processed_event

    #async with buffer_lock:
    startFrom = len(request_buffer)
    request_buffer.extend(new_reqs)
    current_buffer_size = len(request_buffer)
    #await asyncio.sleep(0)

    '''
    if current_buffer_size >= batch_size:
        await process_batch()
    else:
        #batch_processed_event.clear()
        pass
    '''

    #await monitor_cl_rotate_points()

    #await asyncio.sleep(0)
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
    if current_size != last_size_checked and last_size_checked == 0:
        last_size_checked = current_size
        last_time_modified = current_time
    # Check if the time threshold has been reached with no size change

    if (current_size > 0 and (current_time - last_time_modified) >= time_threshold): # and batchCycle < curBatchCycle:
        print("10 ms have passed with no change in buffer size. Processing batch... ", current_size)
        batchCycle = curBatchCycle
        await process_batch()
        last_size_checked = 0  # Reset after processing
        last_time_modified = current_time

async def monitor_cl_rotate_points_loop():
    while True:
        await monitor_cl_rotate_points()
        await asyncio.sleep(time_threshold)  # Sleep for 1 millisecond to prevent high CPU usage

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
        #self.val = np.array(val, dtype=np.float32)
        self.val = val
        self.transformed = None
        self.ignore = False
        self.commandPos = -1

    @property
    def x(self):
        return float(self.val[0])

    @x.setter
    def x(self, value):
        self.val[0] = value

    @property
    def y(self):
        return float(self.val[1])

    @y.setter
    def y(self, value):
        self.val[1] = value

    @property
    def z(self):
        return float(self.val[2])

    @z.setter
    def z(self, value):
        self.val[2] = value

    def __sub__(self, other):
        return Coordinate([self.x - other.x, self.y - other.y, self.z - other.z])
        return Coordinate(self.val-other.val)

    def __add__(self, other):
        return Coordinate([self.x + other.x, self.y + other.y, self.z + other.z])
        return Coordinate(self.val + other.val)

    def __mul__(self, other):
        return Coordinate([self.x * other, self.y * other, self.z * other])
        return Coordinate(self.val * other)

    def __truediv__(self, other):
        return Coordinate([self.x / other, self.y / other, self.z / other])
        return Coordinate(self.val / other)

    def isZero(self):
        return self.x == 0 and self.y == 0 and self.z == 0

    def distance(self, other):
        return math.sqrt(((self.x-other.x)**2)+((self.y-other.y)**2)+((self.z-other.z)**2))

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


async def transformChild(child, position, rotation, totRotation):
    if child.ignore:
        return position, child

    if isinstance(child, Coordinate):
        cPos = child  # if child.transformed is None else child.transformed
    else:
        cPos = child.position  # if child.transformed is None else child.transformed

    rel = cPos * 1
    rel += position
    if isinstance(child, Group):
        rel = await child.transform(rel, rotation)

    if rel.isZero() or totRotation.isZero():
        return rel, child

    # if child.transformed is not None and totRotation.val == child.lastRotation.val:
    #    return child.transformed

    rel = await cl_rotatePoints([[rel.val, totRotation.val]])
    return rel[0], child

def run_async_in_executor(func, *args):
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    try:
        result = new_loop.run_until_complete(func(*args))
    finally:
        new_loop.close()
    return result

executor = None
num_cores = os.cpu_count()
if __name__ == '__main__':
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cores)

class Group(Point3D):
    def __init__(self):
        super().__init__()
        self.children = []
        self.texture = None
        self.ignore = False
        self.commandPos = -1

    def add(self, obj):
        self.children.append(obj)
        obj.parent = self

    def avgPosition(self):
        res = Coordinate()
        num = 0
        for child in self.children:
            if isinstance(child, Coordinate):
                res += child
            else:
                res += child.position
            num += 1
        res /= num
        return res

    def transformCommands(self, parent=-1, cmds=None, level=0):
        if cmds is None:
            cmds = []

        cmd = [self.position, self.rotation, parent, level]
        pos = len(cmds)
        self.commandPos = pos
        cmds.append(cmd)

        for child in self.children:
            if isinstance(child, Group):
                child.transformCommands(pos, cmds, level+1)
            else:
                child.commandPos = len(cmds)
                cmds.append([child, Coordinate(), pos, level+1])

        return cmds

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
        #res = [await process_child(child) for child in self.children]
        #tasks = [process_child(child) for child in self.children]

        res = []

        if True:
            if len(self.children) > 32:
                tasks = []
                waitFor = []
                for child in self.children:
                    tasks.append(transformChild(child, position, rotation, totRotation))
                    if len(tasks) > 8:
                        r = asyncio.gather(*tasks)
                        waitFor.append(r)
                        tasks = []
                        #await asyncio.sleep(0)
                r = asyncio.gather(*tasks)
                waitFor.append(r)
                for wait in waitFor:
                    r = await wait
                    res.extend(r)
            else:
                res = []
                for child in self.children:
                    r = await transformChild(child, position, rotation, totRotation)
                    res.append(r)
        else:
            # Use ProcessPoolExecutor to execute tasks on multiple cores
            loop = asyncio.get_running_loop()
            # Prepare tasks to be submitted to the executor
            tasks = []
            for child in self.children:
                tasks.append(loop.run_in_executor(executor, run_async_in_executor, transformChild, child, position, rotation, totRotation))
            # Await the results of these tasks
            res = await asyncio.gather(*tasks)

        '''
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        with Pool(processes=os.cpu_count()) as pool:
            # Running CPU-bound tasks in separate processes
            tasks = []
            for child in self.children:
                args = functools.partial(run_async_in_executor, transformChild, child, position, rotation, totRotation)
                task = loop.run_in_executor(executor, args)
                tasks.append(task)

            res = await asyncio.gather(*tasks)
        '''
        #res = await cl_rotatePoints(reqs)

        for i, child in enumerate(self.children):
            rel, pchild = res[i]

            child.transformed = rel

            if False and not isinstance(child, Coordinate):
                child.children = pchild.children
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

    def listVertices(self):
        res = []
        for child in self.children:
            if isinstance(child, Coordinate):
                res.append(child)
            else:
                res += child.listVertices()

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
        if image.mode != 'RGB':
            image = image.convert('RGB')  # Convert to RGBA if not already in this mode

        self.image = image

        if False:
            # Convert the PIL image to a string buffer and then to a Pygame surface
            raw_str = image.tobytes("raw", 'RGBA')
            image_size = image.size

            # Create a Pygame Surface
            pygame_surface = pygame.image.fromstring(raw_str, image_size, 'RGBA')

            self.texture = pygame_surface
            self.texture_array = pygame.surfarray.array3d(self.texture)
        else:
            self.texture = image
            self.texture_array = np.array(image)

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

def calculate_z(x, y, vertices):
    # Unpack vertices
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = vertices

    # Vectors AB and AC
    AB = np.array([x2 - x1, y2 - y1, z2 - z1])
    AC = np.array([x3 - x1, y3 - y1, z3 - z1])

    # Cross product to find the normal vector
    n = np.cross(AB, AC)

    # Plane equation coefficients
    a, b, c = n
    d = - (a * x1 + b * y1 + c * z1)

    if c == 0:
        return 9999999 # babbè

    # Solve for z
    z = - (a * x + b * y + d) / c
    return z


async def apply_texture(child, mesh, screen_width, screen_height, ignore_area=None):
    texture_array = mesh.texture_array

    width = mesh.image.width
    height = mesh.image.height

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

    range_width = max_x - min_x
    range_height = max_y - min_y

    screen_array = np.zeros(shape=(range_width, range_height, 3))
    screen_depth = np.zeros(shape=(range_width, range_height))
    vertices = [child.screenVertices[0].val, child.screenVertices[1].val, child.screenVertices[2].val]

    # Unpack vertices
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = vertices

    # Vectors AB and AC
    AB = np.array([x2 - x1, y2 - y1, z2 - z1])
    AC = np.array([x3 - x1, y3 - y1, z3 - z1])

    # Cross product to find the normal vector
    n = np.cross(AB, AC)

    # Plane equation coefficients
    a, b, c = n
    d = - (a * x1 + b * y1 + c * z1)

    if c == 0:
        c = 0.0001

    for y in range(min_y, max_y + 1):
        dy = y - min_y

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
        xx0 = int(xx[0])

        yy = (y - child.drawRange[1][0]) / child.drawRange[1][1]
        yy = int(yy*(height-1))

        z = 0

        diff = math.ceil(xx[1] - xx[0])
        if diff > 0:
            x1 = ((xx[0] - child.drawRange[0][0]) / child.drawRange[0][1])*(width-1)
            x2 = ((xx[1] - child.drawRange[0][0]) / child.drawRange[0][1])*(width-1)
            xInc = (x2-x1) / (diff)

            for i in range(0, int(diff)):
                x = i+xx0
                dx = (i+xx0) - min_x

                if not (0 <= dx < range_width and 0 <= dy < range_height):
                    continue

                if x1 > 0 and x1 < texture_array.shape[0] and yy > 0 and yy <  texture_array.shape[1] and 0 < x < screen_width and 0 < y < screen_height:

                    if i % 10 == 0:
                        z = - (a * x + b * y + d) / c

                    if ignore_area[x, y] > z:
                        continue

                    screen_array[dx, dy] = texture_array[int(x1), yy]
                    screen_depth[dx, dy] = z #calculate_z(x, y, vertices)
                    x1 += xInc

    return screen_array, screen_depth, [[int(min_x), int(max_x)], [int(min_y), int(max_y)]]

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


def calculate_euler_angles(P1, P2):
    # Extract coordinates
    x1, y1, z1 = P1.val
    x2, y2, z2 = P2.val

    # Direction vector from P1 to P2
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1

    # Magnitude of the direction vector
    magnitude = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # Normalize the direction vector
    dx, dy, dz = dx / magnitude, dy / magnitude, dz / magnitude

    # Yaw (phi) - angle from the x-axis in the x-y plane
    yaw = np.arctan2(dy, dx)

    # Pitch (theta) - angle from the z-axis
    pitch = np.arccos(dz)  # acos gives the angle between the vector and the z-axis

    # Roll (psi) - not determinable from a single vector without additional constraints, set to zero
    roll = 0

    # Convert radians to degrees
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    roll_deg = np.degrees(roll)

    return roll_deg, pitch_deg, yaw_deg

class Camera(Group):
    def __init__(self):
        super().__init__()
        self.fov = 1
        self.cmds = None

    async def render(self, scene, pygame, screen):
        width = screen.get_width()
        height = screen.get_height()

        avgFov = ((width+height)/2)
        fov = self.fov

        list = scene.list()

        if False: # check for rendering ingnore checking
            for obj in list:
                pos = obj
                if not isinstance(obj, Coordinate):
                    pos = obj.position
                if isinstance(obj, Group):
                    pos = obj.avgPosition()
                versus = Coordinate(calculate_euler_angles(self.position, pos))

                dmx = ((self.rotation.z%360)-versus.x) % 360
                dmy = ((self.rotation.y%360)-versus.y) % 360
                dmy = 360 - dmy

                if dmx > 180:
                    dmx = 360 - dmx
                if dmy > 180:
                    dmy = 360 - dmx

                obj.section = (0 if dmx < 0 else 1) + (0 if dmy < 0 else 2)
                dist = math.fabs(dmx - dmy)

                obj.ignore = dist > 45 * fov

        #scene.reset()
        #await scene.transform(self.position, self.rotation)

        if self.cmds is None:
            self.cmds = scene.transformCommands()
        cmds = self.cmds
        res = synchronous_cl_commands(cmds, self.position, self.rotation)

        for obj in scene.listVertices():
            if obj.commandPos >= 0:
                obj.transformed = res[obj.commandPos]

        midWidth = width/2
        midHeight = height/2

        def posToScreen(pos):
            diffZ = pos.z
            if diffZ != 0:
                diffZ = fov / diffZ
                #diffZ /= fov
                pos.x *= diffZ
                pos.y *= diffZ
                pos.z *= diffZ

                x = (pos.x * avgFov) + midWidth
                y = (pos.y * avgFov) + midHeight

                if math.isinf(x) or math.isinf(y): # useless
                    return Coordinate()

                x = int(x)
                y = int(y)

                return Coordinate([x, y, pos.z])

            return Coordinate()

        drawTexture = []
        for obj in list:
            #if obj.transformed is None:
            #    continue

            if obj.ignore:
                continue

            if isinstance(obj, Triangle):
                vertices = []
                somethingInside = False
                for vertex in obj.vertices:
                    if vertex.transformed.z > 0:
                        somethingInside = False
                        break

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

        tasks = []
        res = []

        ignoreArea = np.full((width, height), 0)

        def calcIgnoreArea(res):
            for r in res:
                z = r[1]
                txtRange = r[2]

                for x in range(0, txtRange[0][1]-txtRange[0][0]):
                    for y in range(0, txtRange[1][1]-txtRange[1][0]):
                        dx = x + txtRange[0][0]
                        dy = y + txtRange[1][0]

                        if dx >= width or dy >= height or dx < 0 or dy < 0:
                            continue

                        zz = z[x, y]
                        if zz > 0:
                            d = ignoreArea[dx, dy]
                            if d == 0 or zz < d:
                                ignoreArea[dx, dy] = zz

        loop = asyncio.get_running_loop()
        for mesh in drawTexture:
            for child in mesh.children:
                if child.drawRange[0][1] == 0 or child.drawRange[1][1] == 0:
                    continue

                if True:
                    if True:
                        tasks.append(apply_texture(child, mesh, width, height, ignoreArea))
                    else:
                        task = loop.run_in_executor(executor, run_async_in_executor, apply_texture, child, mesh, width, height, ignoreArea)
                        tasks.append(task)

                    if len(tasks) >= num_cores*8:
                        print("starting rendering textures")
                        r = await asyncio.gather(*tasks)
                        calcIgnoreArea(r)
                        print("textures rendered")
                        res.extend(r)
                        tasks = []

                else:
                    r = await apply_texture(child, mesh, width, height, ignoreArea)
                    calcIgnoreArea(r)
                    res.append(r)

        if len(tasks) > 0:
            r = await asyncio.gather(*tasks)
            res.extend(r)

        depth = np.zeros((width, height))
        i = 0
        for r in res:
            s = r[0]
            z = r[1]
            txtRange = r[2]

            for x in range(0, txtRange[0][1]-txtRange[0][0]):
                for y in range(0, txtRange[1][1]-txtRange[1][0]):
                    dx = x + txtRange[0][0]
                    dy = y + txtRange[1][0]

                    #if dx >= width or dy >= height or dx < 0 or dy < 0:
                    #    continue

                    zz = z[x,y]
                    if zz > 0:
                        d = depth[dx, dy]
                        if d == 0 or zz < d:
                            depth[dx,dy] = zz
                            screen_array[dx, dy] = s[x, y]
            i += 1

        new_surface = pygame.surfarray.make_surface(screen_array)
        screen.blit(new_surface, (0, 0))

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
        mesh.loadModelTxt('pokemon.txt')
        #mesh.loadModelTxt('model.txt')
        #mesh.setTexture(Image.open('rainbow.jpeg'))
        scene.add(mesh)

        if False:
            triangle = Triangle()
            triangle.vertices[0] = Coordinate([0, 1, 0])
            triangle.vertices[1] = Coordinate([-1, 0, 0])
            triangle.vertices[2] = Coordinate([1, 0, 0])

            triangleMesh = Mesh()
            triangleMesh.add(triangle)
            triangleMesh.setTexture(Image.open('rainbow.jpeg'))
            scene.add(triangleMesh)

    camera = Camera()
    camera.position.z = -10

    # Initialize Pygame
    pygame.init()

    # Window size
    width, height = 1280, 720
    screen = pygame.display.set_mode((width, height))

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
                #mesh.rotation.z -= moveBy * 5
                #camera.fov += 0.1
            elif keyPressing == pygame.K_RIGHT:
                camera.rotation.z += moveBy * 5
                #mesh.rotation.z += moveBy * 5
                #camera.fov -= 0.1
            elif keyPressing == pygame.K_a:
                mesh.rotation.z -= moveBy * 5
            elif keyPressing == pygame.K_d:
                mesh.rotation.z += moveBy * 5


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
    task1 = asyncio.ensure_future(monitor_cl_rotate_points_loop())
    task2 = asyncio.ensure_future(main())

    # You can wait for the specific tasks if necessary, or just run indefinitely
    await asyncio.gather(task1, task2)

if __name__ == '__main__':
    asyncio.run(run())