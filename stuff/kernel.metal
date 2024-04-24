#include <metal_stdlib>
using namespace metal;

// Utility function to convert degrees to radians (Metal uses radians for trigonometric functions)
inline float radians(float degrees) {
    return degrees * (M_PI / 180.0f);
}

kernel void rotatePoints(device float *coords,
                         device float *rotations,
                         device float *results,
                         constant int &num_points,
                         uint id [[thread_position_in_grid]]) {
    if (id >= num_points) return;

    // Each point has x, y, z values, so index should be 3 times the point index
    int idx = id * 3;
    float x = coords[idx];
    float y = coords[idx + 1];
    float z = coords[idx + 2];

    // Rotations are passed as x, y, z for each point
    float rotation_x = rotations[idx];
    float rotation_y = rotations[idx + 1];
    float rotation_z = rotations[idx + 2];

    float radius, angle, sin_theta, cos_theta;

    // Perform rotation around the X-axis
    if (rotation_x != 0) {
        radius = sqrt(y * y + z * z);
        angle = atan2(z, y) + radians(rotation_x);
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        y = radius * cos_theta;
        z = radius * sin_theta;
    }

    // Perform rotation around the Y-axis
    if (rotation_y != 0) {
        radius = sqrt(x * x + z * z);
        angle = atan2(z, x) + radians(rotation_y);
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        x = radius * cos_theta;
        z = radius * sin_theta;
    }

    // Perform rotation around the Z-axis
    if (rotation_z != 0) {
        radius = sqrt(x * x + y * y);
        angle = atan2(y, x) + radians(rotation_z);
        sin_theta = sin(angle);
        cos_theta = cos(angle);
        x = radius * cos_theta;
        y = radius * sin_theta;
    }

    // Save results
    results[idx] = x;
    results[idx + 1] = y;
    results[idx + 2] = z;
}
