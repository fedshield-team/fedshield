---
name: Python dependency installation
description: Environment-specific guidance for installing this project's runtime dependencies.
---

The project-level Python package installer may include unrelated optional imports from the repository, including Windows service and desktop webview packages. Those packages can fail on the Linux Replit environment even when the API and ML dependencies are valid.

**Why:** A full dependency resolution attempt was blocked by a desktop `webview` build requiring GTK/WebKit libraries, while the application’s core Torch, NumPy, scikit-learn, Flower, FastAPI, SHAP, and authentication dependencies installed successfully when targeted.

**How to apply:** When setup is blocked by unrelated optional imports, install only the runtime packages needed for the requested validation or workflow, and report any optional capture/service limitations rather than changing the project to accommodate them.