import _bootstrap  # noqa: F401

from unittest.mock import patch

from utils import keep_awake

AWAKE = keep_awake._ES_CONTINUOUS | keep_awake._ES_SYSTEM_REQUIRED


def test_should_request_and_release_the_awake_state_on_windows():
    with patch.object(keep_awake.sys, "platform", "win32"), \
            patch.object(keep_awake, "_set_execution_state", return_value=True) as state:
        with keep_awake.keep_system_awake() as active:
            assert active
            state.assert_called_once_with(AWAKE)

    assert [call.args[0] for call in state.call_args_list] == [AWAKE, keep_awake._ES_CONTINUOUS]


def test_should_release_the_awake_state_when_the_job_fails():
    with patch.object(keep_awake.sys, "platform", "win32"), \
            patch.object(keep_awake, "_set_execution_state", return_value=True) as state:
        try:
            with keep_awake.keep_system_awake():
                raise ValueError("job failed")
        except ValueError:
            pass

    assert state.call_args_list[-1].args[0] == keep_awake._ES_CONTINUOUS


def test_should_do_nothing_on_other_systems_or_when_refused():
    with patch.object(keep_awake.sys, "platform", "linux"), \
            patch.object(keep_awake, "_set_execution_state") as state:
        with keep_awake.keep_system_awake() as active:
            assert not active
    state.assert_not_called()

    with patch.object(keep_awake.sys, "platform", "win32"), \
            patch.object(keep_awake, "_set_execution_state", side_effect=OSError) as state:
        with keep_awake.keep_system_awake() as active:
            assert not active
    assert state.call_count == 1
