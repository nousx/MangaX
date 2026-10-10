import _bootstrap  # noqa: F401

from types import SimpleNamespace

from PyQt6 import sip
from PyQt6.QtCore import QParallelAnimationGroup, QPropertyAnimation

from ui.widgets.wheel_filter import _stop_popup_animation


def test_stop_popup_animation_tolerates_deleted_qt_animation():
    animation = QPropertyAnimation()
    menu = SimpleNamespace(aniManager=SimpleNamespace(ani=animation))
    sip.delete(animation)

    assert sip.isdeleted(animation)
    _stop_popup_animation(menu)


def test_stop_popup_animation_tolerates_deleted_animation_group():
    # A deleted animation group raises as soon as it is truth-tested.
    group = QParallelAnimationGroup()
    menu = SimpleNamespace(aniManager=SimpleNamespace(aniGroup=group, ani=None))
    sip.delete(group)

    assert sip.isdeleted(group)
    _stop_popup_animation(menu)
