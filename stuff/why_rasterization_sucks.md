ChatGPT:

Yes. Looking at the texture path specifically, I think OpenCL is almost a distraction here: **most of the texture-rendering cost is coming from the CPU rasterizer architecture**, and several parts multiply the amount of memory traffic/work per visible pixel.

The biggest issue I see is that you're effectively doing a **triangle → temporary framebuffer → temporary depthbuffer → global depthbuffer → final framebuffer** pipeline, instead of rasterizing directly into one framebuffer/depthbuffer.

Your `numba_apply_texture()` allocates two arrays **for every triangle**:

```python
screen_array = np.zeros((range_width, range_height, 3), dtype=np.float32)
screen_depth = np.zeros((range_width, range_height), dtype=np.float32)
```

where `range_width × range_height` is the triangle's **bounding rectangle**, not its actual covered area. :chatgpt-content-reference{index="0"}

So imagine a skinny diagonal triangle whose bounding box is 800×600 but which covers only 20,000 pixels. You're allocating/zeroing:

- 480,000 × 3 floats for color
- 480,000 floats for depth

...to eventually write perhaps 20k pixels.

Then later those temporary arrays are walked **again** to merge them into the real screen/depth buffers.

That is probably your #1 architectural inefficiency.

### 1. You're rendering every triangle into its own mini-framebuffer

The ideal software-rasterizer organization is much closer to:

```text
Triangle
    ↓
rasterize covered pixels
    ↓
depth test global Z buffer
    ↓
write directly to global framebuffer
```

Yours is approximately:

```text
Triangle
    ↓
find bounding rectangle
    ↓
allocate triangle RGB buffer
allocate triangle Z buffer
    ↓
rasterize
    ↓
return buffers
    ↓
scan whole returned bounding rectangle again
    ↓
update ignoreArea
    ↓
scan buffers AGAIN later
    ↓
update final framebuffer/depth buffer
```

This is a huge memory-bandwidth penalty.

And because your temporary buffers use `float32` RGB, you're using **12 bytes/pixel** merely for color instead of 3 or 4 bytes. :chatgpt-content-reference{index="1"}

For a dummy renderer, `uint8` RGB/RGBA would already cut that temporary color bandwidth by ~4×.

---

## 2. There is an especially expensive cache-layout problem

This is subtle but important.

You allocate:

```python
screen_array[range_width, range_height, 3]
```

and then iterate:

```python
for y:
    ...
    for i:
        x = ...
        screen_array[dx, dy] = ...
```

:chatgpt-content-reference{index="2"} :chatgpt-content-reference{index="3"}

NumPy arrays are normally C-order, so the **last dimension is contiguous, then `dy`, then `dx`**.

For:

```text
(width, height, RGB)
```

memory looks roughly like:

```text
[x0,y0]
[x0,y1]
[x0,y2]
...
[x1,y0]
```

But your inner raster loop changes **x**, not y.

So consecutive pixels across a horizontal scanline are jumping by approximately:

```text
range_height * 3 * sizeof(float32)
```

bytes.

At 720 pixels height:

```text
720 × 3 × 4 ≈ 8.6 KB
```

between adjacent horizontally rendered pixels.

That's catastrophically unfriendly to CPU caches.

A conventional image layout should instead be:

```python
screen_array[height, width, 3]
screen_array[y, x]
```

so a scanline writes:

```text
RGB RGB RGB RGB RGB RGB...
```

sequentially.

Pygame's `surfarray` convention being `(width,height,channels)` probably encouraged this representation, but **your rasterizer does not need to use the same internal layout as pygame**.

This alone could produce a surprisingly large speedup.

---

## 3. Your texture array appears to have the opposite coordinate convention

This is potentially both a correctness bug and a performance problem.

You create the texture using:

```python
self.texture_array = np.array(image)
```

:chatgpt-content-reference{index="4"}

For a PIL image, that's conventionally:

```python
texture_array[y, x, channel]
```

with shape:

```text
(height, width, 3)
```

But later you access it as:

```python
texture_array[int(x1), yy]
```

and test:

```python
x1 < texture_array.shape[0]
yy < texture_array.shape[1]
```

:chatgpt-content-reference{index="5"}

So you're effectively interpreting it as:

```python
texture_array[x, y]
```

That's transposed.

Apart from incorrect UV orientation for non-square textures, this also means you're sampling memory in a pretty bad pattern.

Either use:

```python
texture_array[yy, int(x1)]
```

or intentionally transpose it once:

```python
texture_array = np.ascontiguousarray(np.array(image).transpose(1, 0, 2))
```

if you want your renderer to remain x-major.

I would instead standardize the renderer itself to conventional:

```text
[y, x, channel]
```

everywhere.

---

## 4. The `asyncio` parallelization isn't really parallelizing the rasterizer

This is another major one.

You have:

```python
async def async_multiple_numba_apply_texture(...):
    return multiple_numba_apply_texture(...)
```

:chatgpt-content-reference{index="6"}

There is no `await`, executor, worker thread, process, or GPU submission inside that function.

Therefore when you later do conceptually:

```python
await asyncio.gather(task1, task2, task3...)
```

each coroutine gets scheduled, but when it enters:

```python
multiple_numba_apply_texture(...)
```

it performs the whole CPU-bound Numba operation before yielding control.

So effectively you get something similar to:

```text
batch 1 runs
batch 1 finishes
batch 2 runs
batch 2 finishes
batch 3 runs
...
```

not:

```text
batch 1 ──────────┐
batch 2 ──────────┼ simultaneous
batch 3 ──────────┘
```

`asyncio` helps with waiting on I/O. It doesn't parallelize CPU computation.

And `multiple_numba_apply_texture()` itself explicitly loops serially:

```python
for i in range(0, len(options)):
    ...
    r = numba_apply_texture(...)
```

:chatgpt-content-reference{index="7"}

So you've created batching without actual parallel rasterization.

A much better Numba implementation would potentially use:

```python
@njit(parallel=True)
```

and `prange`, although **parallelizing by triangle introduces framebuffer/depth-buffer races** if triangles write directly to the final screen.

That leads naturally to tiled rendering, which I'll come back to.

---

## 5. You're doing dynamic list construction + sorting for every scanline

For every Y coordinate of every triangle:

```python
xx = List()

if ...
    xx.append(...)
if ...
    xx.append(...)
if ...
    xx.append(...)

...
xx.sort()
```

:chatgpt-content-reference{index="8"}

This is logically simple, but computationally ugly.

A triangle has exactly three edges. You don't need a dynamic typed list and general-purpose sort every scanline.

You can classify vertices once:

```text
top
middle
bottom
```

and walk two edges incrementally:

```text
x_left  += dxdy_left
x_right += dxdy_right
```

This is classic scanline rasterization.

Then each row becomes approximately:

```python
for y:
    x_start = ...
    x_end = ...

    for x in range(x_start, x_end):
        ...
```

No:

- `List()`
- append
- deletion
- sorting
- repeated line-equation solving.

That is dramatically easier for Numba to optimize.

---

## 6. You're computing intersections using division rather than incrementally walking edges

Each row calls:

```python
numba_x_from_y(m, b, y)
```

which performs essentially:

```python
x = (y - b) / m
```

:chatgpt-content-reference{index="9"}

That means several floating-point divisions per scanline.

But edge position changes linearly.

You can compute once:

```python
dxdy = (x2 - x1) / (y2 - y1)
```

then:

```python
x += dxdy
```

every row.

Modern CPUs still strongly prefer an addition over repeated division.

---

## 7. Depth calculation isn't incremental either

You correctly derive the triangle plane:

```text
ax + by + cz + d = 0
```

and therefore:

```python
z = -(a*x + b*y + d) / c
```

:chatgpt-content-reference{index="10"}

But then inside the inner loop:

```python
if i % 4 == 0:
    z = -(a * x + b * y + d) / c
```

:chatgpt-content-reference{index="11"}

This has two problems.

First, it's only recalculated every four pixels, so those four pixels share a depth value.

Second, you don't actually need the division in the pixel loop at all.

Because the plane is linear:

\[
z(x+1,y)-z(x,y)=-\frac{a}{c}
\]

Therefore:

```python
z = z_at_left_edge
dzdx = -a / c

for x:
    ...
    z += dzdx
```

One addition per pixel.

And moving down a row:

\[
dzdy=-\frac{b}{c}
\]

This is exactly the sort of thing old-school software renderers exploited heavily.

---

## 8. Texture coordinates should also be incremental

Right now you're deriving texture sampling from the projected bounding rectangle:

```python
yy = (y - drawRange[1][0]) / drawRange[1][1]
...
x1 = ((xx[0] - drawRange[0][0]) / drawRange[0][1]) * (width-1)
x2 = ...
xInc = ...
```

:chatgpt-content-reference{index="12"}

Then:

```python
texture_array[int(x1), yy]
x1 += xInc
```

:chatgpt-content-reference{index="13"}

Incrementing `x1` is good, but the underlying mapping is not really triangle texture mapping. It's mapping the texture through the triangle's **screen-space bounding box**.

A proper triangle rasterizer would associate:

```text
vertex A → (uA, vA)
vertex B → (uB, vB)
vertex C → (uC, vC)
```

and interpolate U/V using barycentric coordinates or gradients.

For perspective correctness you'd interpolate:

\[
\frac{u}{z},\frac{v}{z},\frac{1}{z}
\]

and recover:

\[
u=\frac{u/z}{1/z}
\]

\[
v=\frac{v/z}{1/z}
\]

Even ignoring visual correctness, that representation gives you extremely efficient incremental rasterization.

---

# 9. You're paying for depth testing multiple times

Your per-triangle function already checks:

```python
if ignore_area is not None and ignore_area[x, y] > z:
    continue
```

:chatgpt-content-reference{index="14"}

Then it stores:

```python
screen_depth[dx, dy] = z
```

:chatgpt-content-reference{index="15"}

But afterward the renderer walks returned depth maps to update `ignoreArea`, and eventually walks all triangle buffers again to establish final visibility.

So the depth information goes through several stages.

In a normal rasterizer it should just be:

```python
if z < zbuffer[y, x]:
    zbuffer[y, x] = z
    framebuffer[y, x] = texture[v, u]
```

That's it.

The framebuffer itself becomes the result.

No `screen_depth` per triangle.
No `screen_array` per triangle.
No `calcIgnoreArea`.
No final `calcScreen`.

---

# 10. Bounding boxes aren't clipped early enough

You calculate:

```python
min_x = np.min(...)
max_x = np.max(...)
min_y = ...
max_y = ...
```

and immediately allocate according to those dimensions. :chatgpt-content-reference{index="16"}

A triangle whose projected coordinates are:

```text
(-2000, -1000)
(  400,   200)
(  500,   300)
```

can cause a huge temporary allocation despite almost all of it being outside the screen.

You want:

```python
min_x = max(0, floor(min_x))
max_x = min(screen_width - 1, ceil(max_x))
min_y = max(0, floor(min_y))
max_y = min(screen_height - 1, ceil(max_y))
```

**before doing any per-pixel work.**

Although ideally there wouldn't be a temporary bounding-box allocation at all.

---

# 11. Your visibility/culling stage could eliminate much more work

Before texture rendering you currently perform fairly lightweight visibility checks on transformed triangles. :chatgpt-content-reference{index="17"}

I don't see a proper cheap backface test before rasterization.

Something as simple as screen-space signed area:

\[
A=(x_1-x_0)(y_2-y_0)-(y_1-y_0)(x_2-x_0)
\]

lets you reject triangles facing away from the camera:

```python
if area <= 0:
    continue
```

depending on winding.

For a closed mesh that can eliminate roughly half the triangles before they ever reach the texture rasterizer.

That's a huge gain for almost zero work.

---

# What I think is actually costing you the most

If I had to rank the non-OpenCL problems in **your particular implementation**, I'd put them approximately like this:

| Problem | Likely impact |
|---|---:|
| Per-triangle RGB + depth temporary allocations | **Very high** |
| Multiple passes over those buffers | **Very high** |
| Bad x/y memory access orientation | **Very high** |
| `asyncio` giving the appearance of CPU parallelism without actual parallelism | **High** |
| Rasterizing entire bounding rectangles / lack of early clipping | **High** |
| Dynamic `List` + sort per scanline | **Medium–high** |
| No strong backface/triangle rejection | **Medium–high** |
| Repeated divisions for edge/depth evaluation | **Medium** |
| Float32 framebuffer for 8-bit texture output | **Medium** |
| Texture indexing transposition | **Correctness + cache issue** |

The OpenCL transforms themselves are actually somewhat orthogonal: you're already doing the heavy texture work through Numba/CPU. OpenCL is involved much more in your geometry transforms, e.g. buffers/kernel execution around the coordinate transformations. :chatgpt-content-reference{index="18"}

So replacing OpenCL with Vulkan/CUDA/etc. **without changing the rasterizer architecture probably wouldn't cure the main texture bottleneck**.

## The architecture I'd use while keeping the spirit of your “dummy renderer”

I wouldn't turn it into OpenGL. I'd preserve the homemade renderer and change the texture stage into:

```text
Transform vertices
       ↓
Projection
       ↓
Near-plane clipping
       ↓
Backface culling
       ↓
Screen bounding box / tile assignment
       ↓
─────────────────────────────────
   parallel rasterization
─────────────────────────────────
       ↓
Global uint8 framebuffer
Global float32 depthbuffer
```

Then the core rasterizer becomes conceptually:

```python
@njit
def draw_triangle(..., framebuffer, depthbuffer):

    # bounding box, clipped to screen
    minx = ...
    maxx = ...
    miny = ...
    maxy = ...

    # triangle edge equations / barycentric increments
    ...

    for y in range(miny, maxy + 1):

        # incremental values at beginning of scanline
        ...

        for x in range(minx, maxx + 1):

            if inside:
                inv_z = ...
                z = 1.0 / inv_z

                if z < depthbuffer[y, x]:
                    u = ...
                    v = ...

                    framebuffer[y, x] = texture[v, u]
                    depthbuffer[y, x] = z

            # additions only
            w0 += dw0dx
            w1 += dw1dx
            w2 += dw2dx
```

Most inner-loop operations then become **adds and comparisons**, which is what you want.

And if you want actual multicore parallelism without race conditions, I'd take one additional step: **tile-based rasterization**.

Split 1280×720 into, say, 32×32 tiles:

```text
┌───┬───┬───┬───┐
│32 │32 │32 │32 │
├───┼───┼───┼───┤
│32 │32 │32 │32 │
├───┼───┼───┼───┤
 ...
```

Assign triangles to overlapping tiles, then:

```python
prange(tile_count)
```

Each worker owns its framebuffer/depth region, so there are no triangle-vs-triangle write races.

That would fit the philosophy of this renderer extremely well and should be **orders of magnitude closer to what the CPU actually likes doing** than the current per-triangle temporary-buffer approach.