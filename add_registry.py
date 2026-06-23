import winreg

value = (
    r"C:\Users\megha\OneDrive\Desktop\fedshield\venv\Scripts\python.exe "
    r"C:\Users\megha\OneDrive\Desktop\fedshield\live_capture.py"
)

key = winreg.OpenKey(
    winreg.HKEY_LOCAL_MACHINE,
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run",
    0, winreg.KEY_SET_VALUE
)
winreg.SetValueEx(key, "FedShieldIDS", 0, winreg.REG_SZ, value)
winreg.CloseKey(key)
print("Registry entry added successfully.")
print(f"Value: {value}")