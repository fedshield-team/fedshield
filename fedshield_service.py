"""FedShield — Windows Service.

Install:
    python fedshield_service.py install

Start:
    python fedshield_service.py start

Stop:
    python fedshield_service.py stop

Remove:
    python fedshield_service.py remove
"""

import os
import subprocess
import sys
import time

import servicemanager
import win32event
import win32service
import win32serviceutil


# ─────────────────────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PYTHON_EXE = os.path.join(
    PROJECT_DIR,
    "venv",
    "Scripts",
    "python.exe"
)

CAPTURE_SCRIPT = os.path.join(
    PROJECT_DIR,
    "live_capture.py"
)

LOG_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

LOG_FILE = os.path.join(
    LOG_DIR,
    "fedshield_service.log"
)


class FedShieldService(
    win32serviceutil.ServiceFramework
):

    _svc_name_ = (
        "FedShieldIDS"
    )

    _svc_display_name_ = (
        "FedShield Intrusion Detection Service"
    )

    _svc_description_ = (
        "Privacy-preserving federated "
        "intrusion detection with live "
        "monitoring and firewall response."
    )

    def __init__(
        self,
        args
    ):

        super().__init__(
            args
        )

        self.stop_event = (
            win32event.CreateEvent(
                None,
                0,
                0,
                None
            )
        )

        self.process = None

    def SvcStop(self):

        self._log(
            "FedShield service stopping..."
        )

        self.ReportServiceStatus(
            win32service.SERVICE_STOP_PENDING
        )

        win32event.SetEvent(
            self.stop_event
        )

        if (
            self.process
            and
            self.process.poll()
            is None
        ):

            self.process.terminate()

            try:

                self.process.wait(
                    timeout=10
                )

            except subprocess.TimeoutExpired:

                self.process.kill()

        self._log(
            "FedShield service stopped."
        )

    def SvcDoRun(self):

        self._log(
            "FedShield service starting..."
        )

        servicemanager.LogInfoMsg(
            f"{self._svc_name_} started"
        )

        self._run_capture()

    def _run_capture(self):

        os.makedirs(
            LOG_DIR,
            exist_ok=True
        )

        if not os.path.isfile(
            PYTHON_EXE
        ):

            self._log(
                "Python executable not found: "
                f"{PYTHON_EXE}"
            )

            return

        if not os.path.isfile(
            CAPTURE_SCRIPT
        ):

            self._log(
                "Capture script not found: "
                f"{CAPTURE_SCRIPT}"
            )

            return

        while (
            win32event.WaitForSingleObject(
                self.stop_event,
                0
            )
            != win32event.WAIT_OBJECT_0
        ):

            self._log(
                f"Starting {CAPTURE_SCRIPT}"
            )

            try:

                with open(
                    LOG_FILE,
                    "a",
                    encoding="utf-8"
                ) as log:

                    self.process = (
                        subprocess.Popen(
                            [
                                PYTHON_EXE,
                                CAPTURE_SCRIPT
                            ],

                            cwd=PROJECT_DIR,

                            stdout=log,

                            stderr=log,

                            creationflags=(
                                subprocess.CREATE_NO_WINDOW
                            )
                        )
                    )

                while (
                    self.process.poll()
                    is None
                ):

                    if (
                        win32event.WaitForSingleObject(
                            self.stop_event,
                            2000
                        )
                        == win32event.WAIT_OBJECT_0
                    ):

                        return

                self._log(
                    "live_capture.py exited "
                    f"with code "
                    f"{self.process.returncode}; "
                    "restarting in 5s"
                )

            except Exception as e:

                self._log(
                    f"Error launching capture: {e}"
                )

            for _ in range(5):

                if (
                    win32event.WaitForSingleObject(
                        self.stop_event,
                        1000
                    )
                    == win32event.WAIT_OBJECT_0
                ):

                    return

    def _log(
        self,
        msg
    ):

        try:

            os.makedirs(
                LOG_DIR,
                exist_ok=True
            )

            with open(
                LOG_FILE,
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{msg}\n"
                )

        except OSError:

            pass


if __name__ == "__main__":

    if len(sys.argv) == 1:

        servicemanager.Initialize()

        servicemanager.PrepareToHostSingle(
            FedShieldService
        )

        servicemanager.StartServiceCtrlDispatcher()

    else:

        win32serviceutil.HandleCommandLine(
            FedShieldService
        )