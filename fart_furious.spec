# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for FART & FURIOUS - Pacman with ML
Bundles the game with all ML features into a standalone app.
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all hidden imports for ML libraries
hiddenimports = [
    # PyTorch
    'torch',
    'torch.nn',
    'torch.optim',
    'torch.nn.functional',
    # NumPy
    'numpy',
    'numpy.core._methods',
    'numpy.lib.format',
    # Scikit-learn (if used)
    'sklearn',
    'sklearn.cluster',
    'sklearn.decomposition',
    # Game modules
    'agents',
    'agents.ghost_agent',
    'agents.multi_agent',
    'ml',
    'ml.maze_gan',
    'ml.neural_pathfinder',
    'ml.ghost_evolution',
    'environments',
    'environments.pacman_env',
    'unsupervised',
    'unsupervised.state_encoder',
    'unsupervised.pattern_learner',
    'training',
    # Pygame
    'pygame',
    'pygame.locals',
    'pygame.mixer',
    'pygame.sndarray',
]

# Data files to include
datas = [
    # Include any saved models or config files
    ('training/config.yaml', 'training'),
]

# Analysis
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary large packages
        'matplotlib',
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        'PIL.ImageTk',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ObserveAgenticGhosts',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI app - no console
    disable_windowed_traceback=False,
    argv_emulation=True,  # For macOS
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)

# macOS app bundle
app = BUNDLE(
    exe,
    name='ObserveAgenticGhosts.app',
    icon=None,  # Add .icns file path here if you have one
    bundle_identifier='com.observe.agenticghosts',
    info_plist={
        'CFBundleName': 'Observe Agentic Ghosts',
        'CFBundleDisplayName': 'Observe: Agentic Ghosts Intelligence',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'NSMicrophoneUsageDescription': 'This app uses the microphone for game sounds.',
    },
)
