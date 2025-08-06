#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境验证脚本
"""
import sys
import os


def test_imports():
    """测试所有必要模块的导入"""
    print("🧪 测试模块导入...")

    modules_info = {
        'sys': None,
        'os': None,
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'PIL.Image': 'Pillow Image',
        'PIL.ImageTk': 'Pillow ImageTk',
        'dlib': 'Dlib',
        'scipy': 'SciPy',
        'scipy.stats': 'SciPy Stats',
        'tkinter': 'Tkinter',
        'tkinter.filedialog': 'Tkinter FileDialog',
        'tkinter.messagebox': 'Tkinter MessageBox',
    }

    success_count = 0
    failed_modules = []

    for module, description in modules_info.items():
        try:
            imported_module = __import__(module)

            # 获取版本信息
            version = "未知版本"
            if hasattr(imported_module, '__version__'):
                version = imported_module.__version__
            elif module == 'cv2':
                version = imported_module.__version__
            elif module == 'PIL':
                version = imported_module.__version__

            desc_text = f" ({description})" if description else ""
            print(f"✅ {module}{desc_text} - {version}")
            success_count += 1

        except ImportError as e:
            failed_modules.append((module, str(e)))
            desc_text = f" ({description})" if description else ""
            print(f"❌ {module}{desc_text} - 导入失败: {e}")

    return success_count, failed_modules


def test_opencv_basic():
    """测试OpenCV基本功能"""
    print("\n🔍 测试OpenCV基本功能...")
    try:
        import cv2
        import numpy as np

        # 创建测试图像
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        print(f"✅ OpenCV基本功能正常")
        print(f"   - 版本: {cv2.__version__}")
        print(f"   - 安装路径: {cv2.__file__}")

        # 检查OpenCV目录内容
        cv2_dir = os.path.dirname(cv2.__file__)
        cv2_files = [f for f in os.listdir(cv2_dir) if not f.startswith('__')]
        print(f"   - OpenCV目录包含 {len(cv2_files)} 个文件")

        return True
    except Exception as e:
        print(f"❌ OpenCV功能测试失败: {e}")
        return False


def test_dlib():
    """测试dlib功能"""
    print("\n👤 测试dlib功能...")
    try:
        import dlib

        # 测试人脸检测器
        detector = dlib.get_frontal_face_detector()
        print("✅ Dlib人脸检测器创建成功")
        print(f"   - 安装路径: {dlib.__file__ if hasattr(dlib, '__file__') else '内置模块'}")

        return True
    except Exception as e:
        print(f"❌ Dlib测试失败: {e}")
        return False


def test_tkinter():
    """测试Tkinter GUI功能"""
    print("\n🖥️  测试Tkinter GUI...")
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        from PIL import ImageTk

        # 创建一个简单的窗口测试
        root = tk.Tk()
        root.withdraw()  # 隐藏窗口

        print("✅ Tkinter GUI组件正常")
        print("✅ PIL.ImageTk集成正常")

        root.destroy()
        return True
    except Exception as e:
        print(f"❌ Tkinter测试失败: {e}")
        return False


def check_environment():
    """检查环境信息"""
    print("🌍 环境信息:")
    print(f"   - Python版本: {sys.version}")
    print(f"   - Python路径: {sys.executable}")
    print(f"   - 平台: {sys.platform}")

    # 检查是否在conda环境中
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"   - Conda环境: {conda_env}")

    # 检查site-packages路径
    import site
    print(f"   - site-packages路径:")
    for path in site.getsitepackages():
        print(f"     {path}")


def main():
    """主函数"""
    print("🧪 Conda环境验证脚本")
    print("=" * 50)

    # 检查环境
    check_environment()
    print()

    # 测试模块导入
    success_count, failed_modules = test_imports()
    print()

    # 测试OpenCV
    opencv_ok = test_opencv_basic()
    print()

    # 测试dlib
    dlib_ok = test_dlib()
    print()

    # 测试Tkinter
    tkinter_ok = test_tkinter()
    print()

    # 总结
    print("📊 验证结果:")
    total_modules = success_count + len(failed_modules)
    print(f"   - 模块导入: {success_count}/{total_modules} 成功")
    print(f"   - OpenCV功能: {'✅' if opencv_ok else '❌'}")
    print(f"   - Dlib功能: {'✅' if dlib_ok else '❌'}")
    print(f"   - Tkinter功能: {'✅' if tkinter_ok else '❌'}")

    if failed_modules:
        print(f"\n❌ 失败的模块:")
        for module, error in failed_modules:
            print(f"   - {module}: {error}")

    # 判断是否可以进行打包
    all_critical_ok = opencv_ok and dlib_ok and tkinter_ok and len(failed_modules) == 0

    if all_critical_ok:
        print(f"\n🎉 环境配置完美！可以进行打包！")
        return True
    else:
        print(f"\n⚠️  环境存在问题，需要修复后再进行打包")
        return False


if __name__ == "__main__":
    main()