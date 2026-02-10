import numpy as np
import pytest

from nav.support.image import (
    next_power_of_2,
    pad_array,
    pad_array_to_power_of_2,
    shift_array,
    unpad_array,
)


def test_shift_array() -> None:
    with pytest.raises(ValueError):
        shift_array(np.zeros((10, 10)), [1])

    arr = np.zeros((10, 10))
    assert shift_array(arr, (0, 0)) is arr

    zeros = np.zeros(5)
    arr2 = zeros + 1
    shift1 = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    shiftn1 = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    assert np.all(shift_array(arr2, (1,)) == shift1)
    assert np.all(shift_array(arr2, (-1,)) == shiftn1)
    assert np.all(shift_array(arr2, (5,)) == zeros)

    zeros_2 = np.zeros((5, 5))
    arr_2 = zeros_2 + 1
    shift01_2 = np.array([shift1, shift1, shift1, shift1, shift1])
    shift0n1_2 = np.array([shiftn1, shiftn1, shiftn1, shiftn1, shiftn1])
    shift10_2 = np.array([zeros, arr2, arr2, arr2, arr2])
    shiftn10_2 = np.array([arr2, arr2, arr2, arr2, zeros])
    shift11_2 = np.array([zeros, shift1, shift1, shift1, shift1])
    assert np.all(shift_array(arr_2, (0, 1)) == shift01_2)
    assert np.all(shift_array(arr_2, (0, -1)) == shift0n1_2)
    assert np.all(shift_array(arr_2, (1, 0)) == shift10_2)
    assert np.all(shift_array(arr_2, (-1, 0)) == shiftn10_2)
    assert np.all(shift_array(arr_2, (1, 1)) == shift11_2)

    arr1d = np.arange(5.0) + 3
    arr2d = np.array([arr1d, arr1d + 1, arr1d + 2])
    arr3d = np.array([arr2d, arr2d + 10, arr2d + 20])
    exp = [
        [
            [20.0, 20.0, 20.0, 20.0, 20.0],
            [20.0, 20.0, 20.0, 20.0, 20.0],
            [20.0, 20.0, 20.0, 20.0, 20.0],
        ],
        [
            [20.0, 20.0, 4.0, 5.0, 6.0],
            [20.0, 20.0, 5.0, 6.0, 7.0],
            [20.0, 20.0, 20.0, 20.0, 20.0],
        ],
        [
            [20.0, 20.0, 14.0, 15.0, 16.0],
            [20.0, 20.0, 15.0, 16.0, 17.0],
            [20.0, 20.0, 20.0, 20.0, 20.0],
        ],
    ]
    ret = shift_array(arr3d, (1, -1, 2), fill=20)
    assert np.all(ret == exp)


def test_pad_array() -> None:
    with pytest.raises(ValueError):
        pad_array(np.zeros((10, 10)), [1])

    arr = np.zeros((10, 10))
    assert pad_array(arr, (0, 0)) is arr

    arr1 = np.array([1, 2, 3, 4, 5])
    pad1 = np.array([0, 0, 1, 2, 3, 4, 5, 0, 0])
    arr2 = np.array([arr1, arr1 + 10])
    zero1 = np.array([0, 0, 0, 0, 0])
    assert np.all(pad_array(arr1, (2,)) == pad1)

    pad2a = np.array([zero1, arr1, arr1 + 10, zero1])
    assert np.all(pad_array(arr2, (1, 0)) == pad2a)

    zero10 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    pad10 = np.array([0, 0, 11, 12, 13, 14, 15, 0, 0])
    pad2b = np.array([zero10, pad1, pad10, zero10])
    assert np.all(pad_array(arr2, (1, 2)) == pad2b)

    pad2b[pad2b == 0] = 100
    assert np.all(pad_array(arr2, (1, 2), fill=100) == pad2b)


def test_unpad_array() -> None:
    with pytest.raises(ValueError):
        unpad_array(np.zeros((10, 10)), [1])

    arr = np.zeros((10, 10))
    assert unpad_array(arr, (0, 0)) is arr

    arr1 = np.array([1, 2, 3, 4, 5])
    unpad1 = np.array([3])
    assert np.all(unpad_array(arr1, (2,)) == unpad1)

    arr2 = np.array([arr1, arr1 + 10, arr1 + 20, arr1 + 30, arr1 + 40, arr1 + 50, arr1 + 60])
    unpad2 = np.array([unpad1 + 10, unpad1 + 20, unpad1 + 30, unpad1 + 40, unpad1 + 50])
    assert np.all(unpad_array(arr2, (1, 2)) == unpad2)


def test_next_power_of_2() -> None:
    assert next_power_of_2(1) == 1
    assert next_power_of_2(2) == 2
    assert next_power_of_2(3) == 4
    assert next_power_of_2(4) == 4
    assert next_power_of_2(5) == 8
    assert next_power_of_2(6) == 8
    assert next_power_of_2(7) == 8
    assert next_power_of_2(8) == 8


def test_pad_array_to_power_of_2() -> None:
    ret = pad_array_to_power_of_2(np.array([[1], [2]]))
    assert np.all(ret[0] == np.array([[1], [2]]))
    assert ret[1] == (0, 0)

    ret = pad_array_to_power_of_2(np.array([[1, 2], [3, 4]]))
    assert np.all(ret[0] == np.array([[1, 2], [3, 4]]))
    assert ret[1] == (0, 0)

    with pytest.raises(ValueError):
        pad_array_to_power_of_2(np.array([[1, 2, 3], [4, 5, 6]]))

    ret = pad_array_to_power_of_2(
        np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8],
                [4, 5, 6, 7, 8, 9],
            ]
        )
    )
    assert np.all(
        ret[0]
        == np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 0],
                [0, 2, 3, 4, 5, 6, 7, 0],
                [0, 3, 4, 5, 6, 7, 8, 0],
                [0, 4, 5, 6, 7, 8, 9, 0],
            ]
        )
    )
    assert ret[1] == (0, 1)

    ret = pad_array_to_power_of_2(
        np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8],
                [4, 5, 6, 7, 8, 9],
                [5, 6, 7, 8, 9, 0],
                [6, 7, 8, 9, 0, 1],
            ]
        )
    )
    assert np.all(
        ret[0]
        == np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4, 5, 6, 0],
                [0, 2, 3, 4, 5, 6, 7, 0],
                [0, 3, 4, 5, 6, 7, 8, 0],
                [0, 4, 5, 6, 7, 8, 9, 0],
                [0, 5, 6, 7, 8, 9, 0, 0],
                [0, 6, 7, 8, 9, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )
    assert ret[1] == (1, 1)


def test_array_zoom() -> None:  # TODO: Implement
    ...


def test_array_unzoom() -> None:  # TODO: Implement
    ...


def test_filter_local_maximum() -> None:  # TODO: Implement
    ...


def test_filter_sub_median() -> None:  # TODO: Implement
    ...


def test_filter_downsample() -> None:  # TODO: Implement
    ...


def test_draw_line() -> None:  # TODO: Implement
    ...


def test_draw_rect() -> None:  # TODO: Implement
    ...


def test_draw_circle() -> None:  # TODO: Implement
    ...
