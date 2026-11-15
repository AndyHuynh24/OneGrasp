import pytest

from touch.data.names import parse_sequence_name


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "p_001_index_press_high_x5_right",
            (1, "index_press_high", "index_press", "high", "x5", 1, "right"),
        ),
        (
            "p_012_press_fingers_no-contact_5x_left",
            (12, "press_fingers_no-contact", "press_fingers", "no-contact", "5x", 1, "left"),
        ),
        (
            "p_003_index_press_rotate-right_x5_2_right",
            (3, "index_press_rotate-right", "index_press", "rotate-right", "x5", 2, "right"),
        ),
        (
            "p_001_calibration_routine_left",
            (1, "calibration_routine", "calibration_routine", "", "", 1, "left"),
        ),
        (
            "p_001_press_cupped_onebyone_high_3x_left",
            (1, "press_cupped_onebyone_high", "press_cupped_onebyone", "high", "3x", 1, "left"),
        ),
        (
            "index_press_left_x5_right",
            (None, "index_press_left", "index_press", "left", "x5", 1, "right"),
        ),
        ("type_sentence_5x_both(1)", (None, "type_sentence", "type_sentence", "", "5x", 2, "both")),
    ],
)
def test_parse_sequence_name(name, expected):
    n = parse_sequence_name(name)
    assert (n.participant, n.gesture, n.family, n.variant, n.reps, n.take, n.hand) == expected


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_sequence_name("not_a_sequence")
