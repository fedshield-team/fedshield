"""
FedShield Windows Service
Runs live_capture.py as a persistent background Windows service.

INSTALL:  python fedshield_service.py install
START:    python fedshield_service.py start
STOP:     python fedshield_service.py stop
REMOVE:   python fedshield_service.py remove

Or use Services (services.msc) to set startup type to Automatic.
"""

import sys
import os
import time
import threading
import subprocess
import servicemanager
import win32event
import win32service
import win32serviceutil

# ── Path to your fedshield project ──────────────────────────────────────────
PROJECT_DIR = r"C:\Users\megha\OneDrive\Desktop\fedshield"
PYTHON_EXE  = os.path.join(PROJECT_DIR, r"venv\Scripts\python.exe")
CAPTURE_SCRIPT = os.path.join(PROJECT_DIR, "live_capture.py")
LOG_FILE    = os.path.join(PROJECT_DIR, r"models\fedshield_service.log")


class FedShieldService(win32serviceutil.ServiceFramework):
    _svc_name_         = "FedShieldIDS"
    _svc_display_name_ = "FedShield Intrusion Detection Service"
    _svc_description_  = (
        "Privacy-preserving federated intrusion detection. "
        "Monitors live network traffic and auto-blocks threats via Windows Firewall."
    )

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.process    = None

    def SvcStop(self):
        self._log("FedShield service stopping...")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self._log("FedShield service stopped.")

    def SvcDoRun(self):
        self._log("FedShield service starting...")
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, "")
        )
        self._run_capture()

    def _run_capture(self):
        """Launch live_capture.py as a subprocess and restart it if it crashes."""
        os.chdir(PROJECT_DIR)

        while True:
            # Check if stop was requested
            if win32event.WaitForSingleObject(self.stop_event, 0) == win32event.WAIT_OBJECT_0:
                break

            self._log(f"Starting live_capture.py: {PYTHON_EXE} {CAPTURE_SCRIPT}")
            try:
                with open(LOG_FILE, "a") as log:
                    self.process = subprocess.Popen(
                        [PYTHON_EXE, CAPTURE_SCRIPT],
                        cwd=PROJECT_DIR,
                        stdout=log,
                        stderr=log,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )

                # Wait for process to finish or stop event
                while True:
                    rc = win32event.WaitForSingleObject(self.stop_event, 2000)
                    if rc == win32event.WAIT_OBJECT_0:
                        # Stop requested
                        return
                    if self.process.poll() is not None:
                        # Process exited — log and restart after 5s
                        self._log(f"live_capture.py exited (code {self.process.returncode}). Restarting in 5s...")
                        break

            except Exception as e:
                self._log(f"Error launching capture: {e}")

            # Wait 5 seconds before restarting (unless stop was requested)
            for _ in range(5):
                if win32event.WaitForSingleObject(self.stop_event, 1000) == win32event.WAIT_OBJECT_0:
                    return

    def _log(self, msg):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        try:
            with open(LOG_FILE, "a") as f:
                f.write(line)
        except Exception:
            pass


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Called by SCM
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(FedShieldService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(FedShieldService)