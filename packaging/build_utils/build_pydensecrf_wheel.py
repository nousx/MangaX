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
    print(f"Command: {' '.join(cmd)}")
    
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
        print(f"❌ Error: {description} failed (exit code: {result.returncode})")
        return False
    
    print(f"✓ {description} done")
    return True


def build_wheel():
    """Build the pydensecrf wheel file"""
    
    # Check the Python version
    print(f"Current Python version: {sys.version}")
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    
    # Create the output folder
    wheels_dir = Path("dist/wheels")
    wheels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary build folder
    build_dir = Path("build/pydensecrf_build")
    if build_dir.exists():
        print(f"Removing the old build folder: {build_dir}")
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
            print(f"⚠️  Warning: the build folder could not be removed completely: {e}")
            print(f"   Trying a fallback build folder...")
            import time
            build_dir = Path(f"build/pydensecrf_build_{int(time.time())}")
    
    build_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Output folder: {wheels_dir.absolute()}")
    print(f"✓ Build folder: {build_dir.absolute()}")
    
    # Install the build tools
    if not run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools", "build"],
        "Install the build tools"
    ):
        return False
    
    # Clone the pydensecrf repository
    repo_url = "https://github.com/lucasb-eyer/pydensecrf.git"
    repo_dir = build_dir / "pydensecrf"
    
    if not run_command(
        ["git", "clone", repo_url, str(repo_dir)],
        "Clone the pydensecrf repository",
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
            "Build the wheel file"
        ):
            return False
        
        # build writes to ./dist by default; move the files to the target folder
        build_dist = repo_dir / "dist"
        print(f"\nTrying to move files from {build_dist} to {wheels_dir}")
        
        if not build_dist.exists():
            print(f"❌ The build output folder does not exist: {build_dist}")
            return False
        
        wheel_files_found = list(build_dist.glob("*.whl"))
        if not wheel_files_found:
            print(f"❌ In {build_dist}, no .whl file was found")
            return False
        
        print(f"Found {len(wheel_files_found)} wheel files")
        for wheel_file in wheel_files_found:
            target_file = wheels_dir / wheel_file.name
            print(f"  Moving: {wheel_file.name}")
            print(f"    from: {wheel_file}")
            print(f"    to: {target_file}")
            try:
                shutil.move(str(wheel_file), str(target_file))
                print(f"  ✓ Moved: {wheel_file.name}")
            except Exception as e:
                print(f"  ❌ Moving failed: {e}")
                return False
        
    finally:
        os.chdir(original_dir)
    
    # List the wheel files that were built
    print(f"\n{'='*60}")
    print("Wheel files produced:")
    print(f"{'='*60}")
    
    wheel_files = list(wheels_dir.glob("*.whl"))
    if not wheel_files:
        print("❌ No produced wheel file was found")
        print(f"   Target folder: {wheels_dir.absolute()}")
        return False
    
    for wheel_file in wheel_files:
        file_size = wheel_file.stat().st_size / 1024  # KB
        print(f"✓ {wheel_file.name} ({file_size:.1f} KB)")
    
    print(f"\n{'='*60}")
    print("✅ Build finished!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Upload the wheel file to a GitHub Release:")
    print(f"   gh release upload <tag> {wheels_dir.absolute()}/*.whl")
    print(f"\n2. Add the download link to requirements.txt:")
    print(f"   # Download the prebuilt wheel from the GitHub Release (Python {python_version})")
    print(f"   # https://github.com/hgmzhn/manga-translator-ui/releases/download/<tag>/pydensecrf-*-{python_version}-*.whl")
    print(f"   git+https://github.com/lucasb-eyer/pydensecrf.git  # fall back to installing from source")
    
    return True


def main():
    """Main function"""
    print("="*60)
    print("pydensecrf wheel build script")
    print("="*60)
    
    # Check for git
    if not shutil.which("git"):
        print("❌ Error: git was not found; install Git first")
        return 1
    
    # Check for a C++ build toolchain
    if sys.platform == "win32":
        if not shutil.which("cl.exe"):
            print("⚠️  Warning: cl.exe (the Microsoft C++ compiler) was not found")
            print("   Make sure Visual Studio Build Tools is installed")
            print("   Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            response = input("\nContinue? (y/n): ")
            if response.lower() != 'y':
                return 1
    
    if build_wheel():
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

