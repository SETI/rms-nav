from typing import Any, cast

import numpy as np
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt

from nav.util.types import NDArrayType, NDArrayFloatType, NPType

import oops


#==============================================================================
#
# array MANIPULATION
#
#==============================================================================

def shift_array(array: NDArrayType[NPType],
                offset_u: int,
                offset_v: int,
                fill: Any = 0) -> NDArrayType[NPType]:
    """Shift an array by an offset, filling with zeros or a specified value.

    A positive offset moves the array to the right (larger u) and down (larger v).
    """

    if offset_u == 0 and offset_v == 0:
        return array

    array = array.copy()
    array = np.roll(array, offset_u, 1)
    array = np.roll(array, offset_v, 0)

    if offset_u != 0:
        if offset_u < 0:
            array[:,offset_u:] = fill
        else:
            array[:,:offset_u] = fill
    if offset_v != 0:
        if offset_v < 0:
            array[offset_v:,:] = fill
        else:
            array[:offset_v,:] = fill

    return array


def pad_array(array: NDArrayType[NPType],
              margin: list[int] | tuple[int, int]) -> NDArrayType[NPType]:
    """Pad an array with a zero-filled (U,V) margin on each edge."""

    if margin[0] == 0 and margin[1] == 0:
        return array
    new_array = np.zeros((array.shape[0]+margin[1]*2,array.shape[1]+margin[0]*2),
                         dtype=array.dtype)
    new_array[margin[1]:margin[1]+array.shape[0],
              margin[0]:margin[0]+array.shape[1], ...] = array
    return new_array


def unpad_array(array: NDArrayType[NPType],
                margin: list[int] | tuple[int, int]) -> NDArrayType[NPType]:
    """Remove a padded margin (U,V) from each edge."""

    if margin[0] == 0 and margin[1] == 0:
        return array
    return array[margin[1]:array.shape[0]-margin[1],
                 margin[0]:array.shape[1]-margin[0], ...]

# def compress_saturated_overlay(overlay):
#     """Compress a 2-D RGB array making each color a single bit."""
#     # Compress an RGB overlay assuming everything is either 0 or 255
#     ret = np.empty((overlay.shape[0]/2, overlay.shape[1]), dtype=np.uint8)
#     ret[:,:] = ( (overlay[ ::2,:,0] > 127) |
#                 ((overlay[ ::2,:,1] > 127) << 1) |
#                 ((overlay[ ::2,:,2] > 127) << 2) |
#                 ((overlay[1::2,:,0] > 127) << 3) |
#                 ((overlay[1::2,:,1] > 127) << 4) |
#                 ((overlay[1::2,:,2] > 127) << 5))
#     return ret

# def uncompress_saturated_overlay(overlay):
#     """Uncompress a 2-D RGB array."""
#     ret = np.empty((overlay.shape[0]*2, overlay.shape[1], 3), dtype=np.uint8)
#     ret[ ::2,:,0] =  overlay & 1
#     ret[ ::2,:,1] = (overlay & 2) >> 1
#     ret[ ::2,:,2] = (overlay & 4) >> 2
#     ret[1::2,:,0] = (overlay & 8) >> 3
#     ret[1::2,:,1] = (overlay & 16) >> 4
#     ret[1::2,:,2] = (overlay & 32) >> 5
#     ret *= 255
#     return ret

# def array_interpolate_missing_stripes(data):
#     """Interpolate missing horizontal data in an array.

#     This routine handles an array that has the right side of some lines missing
#     due to data transmission limitations. A pixel is interpolated if it is
#     missing (zero) and the pixels immediately above and below are present
#     (not zero)."""
#     zero_mask = (data == 0.)
#     data_up = np.zeros(data.shape)
#     data_up[:-1,:] = data[1:,:]
#     data_down = np.zeros(data.shape)
#     data_down[1:,:] = data[:-1,:]
#     up_mask = (data_up != 0.)
#     down_mask = (data_down != 0.)
#     good_mask = np.logical_and(zero_mask, up_mask)
#     good_mask = np.logical_and(good_mask, down_mask)
#     data_mean = (data_up+data_down)/2.
#     ret_data = data.copy()
#     ret_data[good_mask] = data_mean[good_mask]
#     return ret_data

def array_zoom(a: NDArrayType[NPType],
               factor: list[int] | tuple[int, ...]) -> NDArrayType[NPType]:
    a = np.asarray(a)
    slices = [slice(0, old, 1/float(f))
                  for (f, old) in zip(factor, a.shape)]
    idxs = (np.mgrid[slices]).astype('i')
    return cast(NDArrayType[NPType], a[tuple(idxs)])


def array_unzoom(array: NDArrayType[NPType],
                 factor: int | list[int] | tuple[int, ...],
                 method: str = 'mean') -> NDArrayType[NPType]:
    """Zoom down by an integer factor. Returned arrays are floating-point."""

    if len(array.shape) == 1:
        assert isinstance(factor, int)
        if factor == 1:
            return array

        shape1 = array.shape
        if shape1[0] % factor != 0:
            raise ValueError('array shape and unzoom factor are incompatible')

        new_shape1 = shape1[0] // factor
        reshaped1 = array.reshape((new_shape1, factor))

        if method == 'min':
            unzoomed = reshaped1.min(axis=1)
        elif method == 'mean':
            unzoomed = reshaped1.mean(axis=1)
        elif method == 'max':
            unzoomed = reshaped1.max(axis=1)
        else:
            raise ValueError(f'Unknown unzoom method "{method}"')
        return cast(NDArrayType[NPType], unzoomed.reshape((new_shape1,)))

    if not isinstance(factor, (tuple, list)):
        assert isinstance(factor, int)
        factor = (factor, factor)

    if factor[0] == 1 and factor[1] == 1:
        return array

    shape2 = array.shape
    if shape2[0] % factor[0] != 0 or shape2[1] % factor[1] != 0:
        raise ValueError('array shape and unzoom factor are incompatible')

    new_shape2 = (shape2[0] // factor[0], shape2[1] // factor[1])
    reshaped2 = array.reshape((new_shape2[0], factor[0], new_shape2[1], factor[1]))

    if method == 'min':
        unzoomed = reshaped2.min(axis=1)
        unzoomed = unzoomed.min(axis=2)
    elif method == 'mean':
        unzoomed = reshaped2.mean(axis=1)
        unzoomed = unzoomed.mean(axis=2)
    elif method == 'max':
        unzoomed = reshaped2.max(axis=1)
        unzoomed = unzoomed.max(axis=2)
    else:
        raise ValueError(f'Unknown unzoom method "{method}"')

    return cast(NDArrayType[NPType], unzoomed)

#==============================================================================
#
# FILTERS
#
#==============================================================================

def filter_local_maximum(data: NDArrayType[NPType],
                         maximum_boxsize: int = 3,
                         median_boxsize: int = 11,
                         maximum_blur: int = 0,
                         maximum_tolerance: float = 1.,
                         minimum_boxsize: int = 0,
                         gaussian_blur: float = 0.) -> NDArrayType[NPType]:
    """Filter an array to find local maxima.

    Process:
        1) Create a mask consisting of the pixels that are local maxima
           within maximum_boxsize
        2) Find the minimum value for the area around each array pixel
           using minimum_boxsize
        3) Remove maxima that are not at least maximum_tolerange times
           the local minimum
        4) Blur the maximum pixels mask to make each a square area of
           maximum_blur X maximum_blur
        5) Compute the median-subtracted value for each array pixel
           using median_boxsize
        6) Copy the median-subtracted array pixels to a new zero-filled
           array where the maximum mask is true
        7) Gaussian blur this final result

    Inputs:
        data                The array
        maximum_boxsize     The box size to use when finding the maximum
                            value for the area around each pixel.
        median_boxsize      The box size to use when finding the median
                            value for the area around each pixel.
        maximum_blur        The amount to blur the maximum filter. If a pixel
                            is marked as a maximum, then the pixels in a
                            blur X blur square will also be marked as a
                            maximum.
        maximum_tolerance   The factor above the local minimum that a
                            maximum pixel has to be in order to be included
                            in the final result.
        minimum_boxsize     The box size to use when finding the minimum
                            value for the area around each pixel.
        gaussian_blur       The amount to blur the final result.
    """

    assert maximum_boxsize > 0
    max_filter = ndimage.maximum_filter(data, maximum_boxsize)
    mask = data == max_filter

    if minimum_boxsize:
        min_filter = ndimage.minimum_filter(data, minimum_boxsize)
        tol_mask = data >= min_filter * maximum_tolerance
        mask = np.logical_and(mask, tol_mask)

    if maximum_blur:
        mask = ndimage.maximum_filter(mask, maximum_blur)

    if median_boxsize:
        flat = data - ndimage.median_filter(data, median_boxsize)
    else:
        flat = data

    filtered = np.zeros(data.shape, dtype=data.dtype)
    filtered[mask] = flat[mask]

    if gaussian_blur:
        filtered = ndimage.gaussian_filter(filtered, gaussian_blur)

    return filtered


_FOOTPRINT_CACHE: dict[int, Any] = {}

def filter_sub_median(data: NDArrayFloatType,
                      median_boxsize: int = 11,
                      gaussian_blur: float = 0.,
                      footprint: str= 'square') -> NDArrayFloatType:
    """Compute the median-subtracted value for each pixel.

    Inputs:
        data                The array
        median_boxsize      The box size to use when finding the median
                            value for the area around each pixel.
        gaussian_blur       The amount to blur the median value before
                            subtracting it from the array.
        footprint           The shape of footprint to use ('square', 'circle').
    """

    if not median_boxsize and not gaussian_blur:
        return data

    sub = data

    if median_boxsize:
        if footprint == 'square':
            sub = ndimage.median_filter(sub, median_boxsize)
        else:
            assert footprint == 'circle'
            if median_boxsize in _FOOTPRINT_CACHE:
                footprint_arr = _FOOTPRINT_CACHE[median_boxsize]
            else:
                footprint_arr = np.zeros((median_boxsize, median_boxsize),
                                         dtype='bool')
                box_half = median_boxsize // 2
                x = (np.tile(np.arange(median_boxsize)-box_half, median_boxsize).
                     reshape((median_boxsize, median_boxsize)))
                y = (np.repeat(np.arange(median_boxsize)-box_half, median_boxsize).
                     reshape((median_boxsize, median_boxsize)))
                dist = np.sqrt(x**2 + y**2)
                footprint_arr[dist <= box_half+.5] = 1
                plt.imshow(footprint_arr)
                plt.show()
                _FOOTPRINT_CACHE[median_boxsize] = footprint_arr
            sub = ndimage.median_filter(sub, footprint=footprint_arr)

    if gaussian_blur:
        sub = ndimage.gaussian_filter(sub, gaussian_blur)

    return data - sub


def filter_downsample(arr: NDArrayFloatType,
                      amt_y: int,
                      amt_x: int) -> NDArrayFloatType:
    assert arr.shape[0] % amt_y == 0
    assert arr.shape[1] % amt_x == 0
    ny = arr.shape[0] // amt_y
    nx = arr.shape[1] // amt_x
    ret = (np.swapaxes(arr.reshape(ny, amt_y, nx, amt_x), 1, 2).
           reshape(ny, nx, amt_x*amt_y).mean(axis=2))
    return cast(NDArrayFloatType, ret)

#==============================================================================
#
# DRAWING ROUTINES
#
#==============================================================================

def draw_line(img: NDArrayType[NPType],
              color: int | float | list[int | float] | tuple[int | float, ...],
              x0: int | float,
              y0: int | float,
              x1: int | float,
              y1: int | float,
              thickness: int = 1) -> None:
    """Draw a line using Bresenham's algorithm with the given thickness.

    The line is drawn by drawing each point as a line perpendicular to
    the main line.

    Input:

        img        The 2-D (or higher) array to draw on.
        x0, y0     The starting point.
        x1, y1     The ending point.
        color      The scalar (or higher) color to draw.
        thickness  The thickness (total width) of the line.
    """

    x0 = int(x0)
    x1 = int(x1)
    y0 = int(y0)
    y1 = int(y1)

    if thickness == 1:
        # Do the simple version
        dx = abs(x1-x0)
        dy = abs(y1-y0)
        if x0 < x1:
            sx = 1
        else:
            sx = -1
        if y0 < y1:
            sy = 1
        else:
            sy = -1
        err = dx-dy

        while True:
            img[y0, x0] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2*err
            if e2 > -dy:
                err = err - dy
                x0 = x0 + sx
            if e2 < dx:
                err = err + dx
                y0 = y0 + sy
        return

    # Find the perpendicular angle
    angle = np.arctan2(y1-y0, x1-x0)
    x_offset = int(np.round(np.cos(angle)))
    y_offset = int(np.round(np.sin(angle)))
    perp_angle = angle + np.pi/2
    perp_x1 = int(np.round(thickness/2.*np.cos(perp_angle)))
    perp_x0 = -perp_x1
    perp_y1 = int(np.round(thickness/2.*np.sin(perp_angle)))
    perp_y0 = -perp_y1
    if perp_x0 == perp_x1 and perp_y0 == perp_y1:
        draw_line(img, color, x0, y0, x1, y1)
        return

    # Compute the perpendicular offsets using one pass of Bresenham's
    perp_offsets_x = []
    perp_offsets_y = []

    dx = abs(perp_x1-perp_x0)
    dy = abs(perp_y1-perp_y0)
    if perp_x0 < perp_x1:
        sx = 1
    else:
        sx = -1
    if perp_y0 < perp_y1:
        sy = 1
    else:
        sy = -1
    err = dx-dy

    while True:
        # There's a better way to do this, but it's patented by IBM!
        # So just do something brute force instead
        perp_offsets_x.append(perp_x0)
        perp_offsets_y.append(perp_y0)
        perp_offsets_x.append(perp_x0+x_offset)
        perp_offsets_y.append(perp_y0)
        perp_offsets_x.append(perp_x0)
        perp_offsets_y.append(perp_y0+y_offset)
        perp_offsets_x.append(perp_x0+x_offset)
        perp_offsets_y.append(perp_y0+y_offset)
        if perp_x0 == perp_x1 and perp_y0 == perp_y1:
            break
        e2 = 2*err
        if e2 > -dy:
            err = err - dy
            perp_x0 = perp_x0 + sx
        if e2 < dx:
            err = err + dx
            perp_y0 = perp_y0 + sy

    # Now draw the final line applying the offsets
    dx = abs(x1-x0)
    dy = abs(y1-y0)
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    if y0 < y1:
        sy = 1
    else:
        sy = -1
    err = dx-dy

    while True:
        for i in range(len(perp_offsets_x)):
            img[y0+perp_offsets_y[i], x0+perp_offsets_x[i]] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2*err
        if e2 > -dy:
            err = err - dy
            x0 = x0 + sx
        if e2 < dx:
            err = err + dx
            y0 = y0 + sy


def draw_rect(img: NDArrayType[NPType],
              color: int | float | list[int | float] | tuple[int | float, ...],
              xctr: int,
              yctr: int,
              xhalfwidth: int,
              yhalfwidth: int,
              thickness: int = 1) -> None:
    """Draw a rectangle with the given line thickness.

    Input:

        img        The 2-D (or higher) array to draw on.
        xctr, yctr The center of the rectangle.
        xhalfwidth The width of the rectangle on each side of the center.
        yhalfwidth This is the inner border of the rectangle.
        color      The scalar (or higher) color to draw.
        thickness  The thickness (total width) of the line.
    """

    # Top
    img[yctr-yhalfwidth-thickness+1:yctr-yhalfwidth+1,
        xctr-xhalfwidth-thickness+1:xctr+xhalfwidth+thickness] = color
    # Bottom
    img[yctr+yhalfwidth:yctr+yhalfwidth+thickness,
        xctr-xhalfwidth-thickness+1:xctr+xhalfwidth+thickness] = color
    # Left
    img[yctr-yhalfwidth-thickness+1:yctr+yhalfwidth+thickness,
        xctr-xhalfwidth-thickness+1:xctr-xhalfwidth+1] = color
    # Right
    img[yctr-yhalfwidth-thickness+1:yctr+yhalfwidth+thickness,
        xctr+xhalfwidth:xctr+xhalfwidth+thickness] = color

def draw_circle(img: NDArrayType[NPType],
                color: int | float | list[int | float] | tuple[int | float, ...],
                x0: int,
                y0: int,
                r: int,
                thickness: int = 1) -> None:
    """Draw a circle using Bresenham's algorithm with the given thickness.

    Input:

        img        The 2-D (or higher) array to draw on.
        x0, y0     The middle of the circle.
        r          The radius of the circle.
        color      The scalar (or higher) color to draw.
        thickness  The thickness (total width) of the circle.
    """

    def _draw_circle(img: NDArrayType[NPType],
                     color: int | float | list[int | float] | tuple[int | float, ...],
                     x0: int,
                     y0: int,
                     r: float | np.floating[Any],
                     bigpixel: bool) -> None:
        x = -r
        y = 0
        err = 2-2*r
        if bigpixel:
            off_list = [-1, 0, 1]
        else:
            off_list = [0]

        while x < 0:
            for xoff in off_list:
                for yoff in off_list:
                    if (0 <= y0+y+yoff < img.shape[0] and
                            0 <= x0-x+xoff < img.shape[1]):
                        img[int(y0+y+yoff), int(x0-x+xoff)] = color
                    if (0 <= y0-x+yoff < img.shape[0] and
                            0 <= x0-y+xoff < img.shape[1]):
                        img[int(y0-x+yoff), int(x0-y+xoff)] = color
                    if (0 <= y0-y+yoff < img.shape[0] and
                            0 <= x0+x+xoff < img.shape[1]):
                        img[int(y0-y+yoff), int(x0+x+xoff)] = color
                    if (0 <= y0+x+yoff < img.shape[0] and
                            0 <= x0+y+xoff < img.shape[1]):
                        img[int(y0+x+yoff), int(x0+y+xoff)] = color
            r = err
            if r <= y:
                y = y+1
                err = err+y*2+1
            if r > x or err > y:
                x = x+1
                err = err+x*2+1

    x0 = int(x0)
    y0 = int(y0)
    r = int(r)

    if thickness == 1:
        _draw_circle(img, color, x0, y0, r, False)
        return

    if thickness <= 3:
        _draw_circle(img, color, x0, y0, r, True)
        return

    # This is not perfect, but it's simple
    for r_offset in np.arange(-(thickness-2)/2., (thickness-2)/2.+0.5, 0.5):
        _draw_circle(img, color, x0, y0, r+r_offset, True)

#==============================================================================
#
# MISC
#
#==============================================================================

# def hsv_to_rgb(data: NDArrayType[NPType]) -> NDArrayType[NPType]:
#     """Convert an array [...,3] of HSV values into RGB values.

#     This is the same as Python's colorsys.hsv_to_rgb but is vectorized.
#     """

#     # From http://www.rapidtables.com/convert/color/hsv-to-rgb.htm
#     # C = V*S
#     # X = C*(1-|(h/60 deg) mod 2 - 1|)
#     # m = V-C

#     deg60 = oops.PI/3
#     h = data[..., 0] * oops.TWOPI
#     c = cast(NDArrayType[NPType], data[..., 1]) * cast(NDArrayType[NPType], data[..., 2])
#     x = c * (1 - np.abs((h / deg60) % 2 - 1))
#     m = data[..., 2] - c

#     ret = np.zeros(data.shape, dtype=data.dtype)

#     is_0_60 = h < deg60
#     is_60_120 = (deg60 <= h) & (h < deg60*2)
#     is_120_180 = (deg60*2 <= h) & (h < deg60*3)
#     is_180_240 = (deg60*3 <= h) & (h < deg60*4)
#     is_240_300 = (deg60*4 <= h) & (h < deg60*5)
#     is_300_360 = deg60*5 <= h

#     cond = is_0_60 | is_300_360
#     ret[cond, 0] = c[cond]
#     cond = is_60_120 | is_240_300
#     ret[cond, 0] = x[cond]

#     cond = is_0_60 | is_180_240
#     ret[cond, 1] = x[cond]
#     cond = is_60_120 | is_120_180
#     ret[cond, 1] = c[cond]

#     cond = is_120_180 | is_300_360
#     ret[cond, 2] = x[cond]
#     cond = is_180_240 | is_240_300
#     ret[cond, 2] = c[cond]

#     ret[..., 0] += m
#     ret[..., 1] += m
#     ret[..., 2] += m

#     return ret
