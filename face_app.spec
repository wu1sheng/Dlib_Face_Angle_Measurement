# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import glob
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 自动检测conda环境路径
conda_env_path = sys.prefix
dll_dir_path = os.path.join(conda_env_path, 'Library', 'bin')
site_packages_path = os.path.join(conda_env_path, 'Lib', 'site-packages')

print(f"🔧 检测到的Conda环境: {conda_env_path}")
print(f"📁 DLL路径: {dll_dir_path}")
print(f"📦 Site-packages路径: {site_packages_path}")

# 收集OpenCV数据
opencv_data = []
try:
    opencv_data = collect_data_files('cv2')
    print(f"✅ 收集到 {len(opencv_data)} 个OpenCV数据文件")
except Exception as e:
    print(f"⚠️ 收集OpenCV数据时出现问题: {e}")

# 收集OpenCV子模块
opencv_submodules = []
try:
    opencv_submodules = collect_submodules('cv2')
    print(f"✅ 收集到 {len(opencv_submodules)} 个OpenCV子模块")
except Exception as e:
    print(f"⚠️ 收集OpenCV子模块时出现问题: {e}")

# 添加dlib的.pyd文件
dlib_pyd_path = os.path.join(site_packages_path, '_dlib_pybind11.cp39-win_amd64.pyd')
binaries = []
if os.path.exists(dlib_pyd_path):
    binaries.append((dlib_pyd_path, '.'))
    print(f"✅ 添加dlib模块: {dlib_pyd_path}")
else:
    print("⚠️ 未找到dlib的pyd模块，可能打包会失败")

# 搜集常用DLL
if os.path.exists(dll_dir_path):
    essential_dll_patterns = [
        'opencv_*.dll',
        '*tbb*.dll',
        'mkl_*.dll',
        '*blas*.dll',
        '*lapack*.dll',
        'libiomp5md.dll',
        'msvcp*.dll',
        'vcruntime*.dll',
        'api-ms-*.dll',
        'concrt*.dll',
        'vcomp*.dll',
    ]
    for pattern in essential_dll_patterns:
        matching_dlls = glob.glob(os.path.join(dll_dir_path, pattern))
        for dll in matching_dlls:
            binaries.append((dll, '.'))

    print(f"✅ 包含 {len(binaries)} 个DLL文件")

# 数据文件
datas = [('model', 'model')]
datas.extend(opencv_data)

# 隐藏导入
hiddenimports = [
    'cv2',
    'numpy',
    'numpy.core',
    'numpy.core._methods',
    'numpy.linalg',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'PIL._imaging',
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    '_tkinter',
    'dlib',
    'scipy',
    'scipy.stats',
]

hiddenimports.extend(opencv_submodules)

a = Analysis(
    ['gui_app.py'],
    pathex=[os.path.abspath('.')],  # 🚫 不要加 site-packages
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'pandas',
        'jupyter',
        'IPython',
        'notebook',
        'pytest',
        'sphinx',
        'numba',
        'sympy',
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
    name='人脸几何特征分析工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

print("🎉 配置完成！")
