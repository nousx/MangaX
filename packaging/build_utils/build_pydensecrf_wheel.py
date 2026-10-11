# -*- coding: utf-8 -*-
"""
Script that builds the pydensecrf wheel file.
Used to compile pydensecrf locally and then upload it to a GitHub Release for users to download

Usage:
1. Make sure a C++ build toolchain is installed
2. Run: python build_utils/build_pydensecrf_wheel.py
3. The wheel file is written to the dist/wheels/ folder
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil

PATH_ROOT = Path(__file__).resolve().parents[2]
if str(PATH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATH_ROOT))

from desktop_qt_ui.core.git_update_helpers import non_interactive_git_env


def run_command(cmd, description, *, env=None):
    """Run a command and show its output"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=False,
        env=env,
    )
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"❌ 错误: {description} 失败 (退出码: {result.returncode})")
        return False
    
    print(f"✓ {description} 完成")
    return True


def build_wheel():
    """Build the pydensecrf wheel file"""
    
    # Check the Python version
    print(f"当前 Python 版本: {sys.version}")
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    
    # Create the output folder
    wheels_dir = Path("dist/wheels")
    wheels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary build folder
    build_dir = Path("build/pydensecrf_build")
    if build_dir.exists():
        print(f"清理旧的构建目录: {build_dir}")
        try:
            # On Windows, files of a Git repository may be locked and need special handling
            if sys.platform == "win32":
                def handle_remove_readonly(func, path, exc):
                    """Handle the error of deleting a read-only file"""
                    import stat
                    if not os.access(path, os.W_OK):
                        os.chmod(path, stat.S_IWUSR)
                        func(path)
                    else:
                        raise
                
                shutil.rmtree(build_dir, onerror=handle_remove_readonly)
            else:
                shutil.rmtree(build_dir)
        except Exception as e:
            print(f"⚠️  警告: 无法完全清理构建目录: {e}")
            print(f"   尝试使用备用构建目录...")
            import time
            build_dir = Path(f"build/pydensecrf_build_{int(time.time())}")
    
    build_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ 输出目录: {wheels_dir.absolute()}")
    print(f"✓ 构建目录: {build_dir.absolute()}")
    
    # Install the build tools
    if not run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools", "build"],
        "安装构建工具"
    ):
        return False
    
    # Clone the pydensecrf repository
    repo_url = "https://github.com/lucasb-eyer/pydensecrf.git"
    repo_dir = build_dir / "pydensecrf"
    
    if not run_command(
        ["git", "clone", repo_url, str(repo_dir)],
        "克隆 pydensecrf 仓库",
        env=non_interactive_git_env(),
    ):

        return False
    
    # Build the wheel
    original_dir = os.getcwd()
    try:
        os.chdir(repo_dir)
        
        # Build the wheel with python -m build
        if not run_command(
            [sys.executable, "-m", "build", "--wheel"],
            "构建 wheel 文件"
        ):
            return False
        
        # build writes to ./dist by default; move the files to the target folder
        build_dist = repo_dir / "dist"
        print(f"\n尝试从 {build_dist} 移动文件到 {wheels_dir}")
        
        if not build_dist.exists():
            print(f"❌ 构建输出目录不存在: {build_dist}")
            return False
        
        wheel_files_found = list(build_dist.glob("*.whl"))
        if not wheel_files_found:
            print(f"❌ 在 {build_dist} 中没有找到 .whl 文件")
            return False
        
        print(f"找到 {len(wheel_files_found)} 个 wheel 文件")
        for wheel_file in wheel_files_found:
            target_file = wheels_dir / wheel_file.name
            print(f"  移动: {wheel_file.name}")
            print(f"    从: {wheel_file}")
            print(f"    到: {target_file}")
            try:
                shutil.move(str(wheel_file), str(target_file))
                print(f"  ✓ 已移动: {wheel_file.name}")
            except Exception as e:
                print(f"  ❌ 移动失败: {e}")
                return False
        
    finally:
        os.chdir(original_dir)
    
    # List the wheel files that were built
    print(f"\n{'='*60}")
    print("生成的 wheel 文件:")
    print(f"{'='*60}")
    
    wheel_files = list(wheels_dir.glob("*.whl"))
    if not wheel_files:
        print("❌ 没有找到生成的 wheel 文件")
        print(f"   目标目录: {wheels_dir.absolute()}")
        return False
    
    for wheel_file in wheel_files:
        file_size = wheel_file.stat().st_size / 1024  # KB
        print(f"✓ {wheel_file.name} ({file_size:.1f} KB)")
    
    print(f"\n{'='*60}")
    print("✅ 构建完成!")
    print(f"{'='*60}")
    print(f"\n下一步:")
    print(f"1. 上传 wheel 文件到 GitHub Release:")
    print(f"   gh release upload <tag> {wheels_dir.absolute()}/*.whl")
    print(f"\n2. 在 requirements.txt 中添加下载链接:")
    print(f"   # 从 GitHub Release 下载预编译的 wheel (Python {python_version})")
    print(f"   # https://github.com/hgmzhn/manga-translator-ui/releases/download/<tag>/pydensecrf-*-{python_version}-*.whl")
    print(f"   git+https://github.com/lucasb-eyer/pydensecrf.git  # fallback 到源码安装")
    
    return True


def main():
    """Main function"""
    print("="*60)
    print("pydensecrf Wheel 构建脚本")
    print("="*60)
    
    # Check for git
    if not shutil.which("git"):
        print("❌ 错误: 未找到 git,请先安装 Git")
        return 1
    
    # Check for a C++ build toolchain
    if sys.platform == "win32":
        if not shutil.which("cl.exe"):
            print("⚠️  警告: 未找到 cl.exe (Microsoft C++ 编译器)")
            print("   请确保已安装 Visual Studio Build Tools")
            print("   下载地址: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            response = input("\n是否继续? (y/n): ")
            if response.lower() != 'y':
                return 1
    
    if build_wheel():
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

