"""Shows whether the Codex CLI is installed and signed in, with a sign-in button."""
import threading

from manga_translator.codex_account import (
    CODEX_INSTALL_URL,
    STATE_ERROR,
    STATE_MISSING,
    STATE_SIGNED_IN,
    CodexStatus,
    codex_status,
    start_codex_login,
)
from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout
from qfluentwidgets import BodyLabel, PrimaryPushButton, PushButton, SimpleCardWidget, StrongBodyLabel

# After the sign-in window opens, re-check until the account appears.
_LOGIN_POLL_INTERVAL_MS = 3000
_LOGIN_POLL_LIMIT = 100


class CodexAccountPanel(SimpleCardWidget):
    _status_ready = pyqtSignal(object)

    def __init__(self, t_func, cli_path_getter, parent=None):
        super().__init__(parent)
        self._t = t_func
        self._cli_path_getter = cli_path_getter
        self._checking = False
        self._polls_left = 0
        self.status = None

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        hint = BodyLabel(self._t("Codex CLI account hint"))
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.status_label = StrongBodyLabel(self._t("Codex status checking"))
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.detail_label = BodyLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail_label.hide()
        layout.addWidget(self.detail_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.login_button = PrimaryPushButton(self._t("Codex sign in"))
        self.login_button.clicked.connect(self._on_login)
        self.install_button = PrimaryPushButton(self._t("Codex get CLI"))
        self.install_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(CODEX_INSTALL_URL)))
        self.refresh_button = PushButton(self._t("Codex check again"))
        self.refresh_button.clicked.connect(self.refresh)
        for button in (self.login_button, self.install_button, self.refresh_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addStretch(1)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(_LOGIN_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._on_poll)
        self._status_ready.connect(self._show_status)

        self.login_button.hide()
        self.install_button.hide()
        self.refresh()

    def _cli_path(self) -> str:
        try:
            return str(self._cli_path_getter() or "")
        except Exception:
            return ""

    def refresh(self):
        """Check the account on a worker thread; `codex login status` can take seconds."""
        if self._checking:
            return
        self._checking = True
        self.refresh_button.setEnabled(False)
        cli_path = self._cli_path()

        def work():
            status = codex_status(cli_path)
            try:
                self._status_ready.emit(status)
            except RuntimeError:
                # The settings page was rebuilt while the check was running.
                pass

        threading.Thread(target=work, name="codex-status", daemon=True).start()

    def _show_status(self, status: CodexStatus):
        self._checking = False
        self.status = status
        self.refresh_button.setEnabled(True)
        signed_in = status.state == STATE_SIGNED_IN
        missing = status.state == STATE_MISSING
        if signed_in:
            self._poll_timer.stop()
            text = self._t("Codex status signed in")
        elif missing:
            text = self._t("Codex status missing")
        elif status.state == STATE_ERROR:
            text = self._t("Codex status error")
        else:
            text = self._t("Codex status signed out")
        self.status_label.setText(text)
        self.detail_label.setText(status.detail)
        self.detail_label.setVisible(bool(status.detail))
        self.install_button.setVisible(missing)
        self.login_button.setVisible(not missing and not signed_in)

    def _on_login(self):
        if not start_codex_login(self._cli_path()):
            self.status_label.setText(self._t("Codex status error"))
            return
        self.status_label.setText(self._t("Codex login opened"))
        self._polls_left = _LOGIN_POLL_LIMIT
        self._poll_timer.start()

    def _on_poll(self):
        self._polls_left -= 1
        if self._polls_left <= 0:
            self._poll_timer.stop()
        self.refresh()
