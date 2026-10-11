#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version check script - merged in from launch.py.
Checks the current version and the remote version
"""
import subprocess
import sys
import os
import argparse
from pathlib import Path

PATH_ROOT = Path(__file__).parent.parent
if str(PATH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATH_ROOT))

from desktop_qt_ui.core.git_update_helpers import non_interactive_git_env

# Git path
def get_git_command():
    """Get the path of the git command"""
    portable_git = PATH_ROOT / "PortableGit" / "cmd" / "git.exe"
    if portable_git.exists():
        return str(portable_git)
    return os.environ.get('GIT', "git")

def get_current_version():
    """Get the current version"""
    version_file = Path(__file__).parent / "VERSION"
    try:
        if version_file.exists():
            return version_file.read_text(encoding='utf-8').strip()
        else:
            return "unknown"
    except Exception as e:
        return "unknown"

def get_remote_version():
    """Get the remote version"""
    git_cmd = get_git_command()
    try:
        # Get the content of the remote VERSION file with git show
        result = subprocess.run(
            [git_cmd, 'show', 'origin/main:packaging/VERSION'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "unknown"
    except Exception:
        return "unknown"

def fetch_remote_refs(git_cmd):
    """Sync the remote references; returns False on failure."""
    try:
        result = subprocess.run(
            [git_cmd, 'fetch', 'origin'],
            capture_output=True,
            text=True,
            check=False,
            env=non_interactive_git_env(),
        )
        return result.returncode == 0
    except Exception:
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='版本检查脚本')
    parser.add_argument('--brief', action='store_true', help='简洁模式：仅显示版本和更新提示')
    parser.add_argument('--export-vars', action='store_true', help='导出环境变量格式（用于bat脚本）')
    args = parser.parse_args()
    
    current_version = get_current_version()
    
    git_cmd = get_git_command()
    fetch_ok = fetch_remote_refs(git_cmd)
    remote_version = get_remote_version() if fetch_ok else "unknown"
    
    # Output in environment variable format (for the bat scripts)
    if args.export_vars:
        print(f"CURRENT_VERSION={current_version}")
        print(f"REMOTE_VERSION={remote_version}")
        return 0
    
    if args.brief:
        # Brief mode - for script 3 (the launcher)
        print("")
        print("========================================")
        print("漫画翻译器 - 启动中")
        print("========================================")
        print(f"当前版本 - {current_version}")
        
        if remote_version == "unknown":
            print("")
            print("[警告] 无法获取远程版本信息,可能网络问题")
            print("")
            return 1
        if current_version != remote_version:
            print(f"远程版本 - {remote_version}")
            print("")
            print("[提示] 发现新版本可用！")
            print("请运行 步骤4-更新维护.bat 进行更新")
            print("")
        else:
            print("")
            print("[信息] 已是最新版本")
            print("")
    else:
        # Detailed mode - for script 4 (update and maintenance)
        print(f"当前版本 - {current_version}")
        print(f"远程版本 - {remote_version}")
        
        # Check whether there is an update
        if remote_version == "unknown":
            print("")
            print("[警告] 无法获取远程版本信息,可能网络问题")
            return 1
        elif current_version == remote_version:
            print("")
            print("[信息] 当前已是最新版本")
            return 0
        else:
            print("")
            print("[发现新版本]")
            print("")
            
            # Try to read the remote CHANGELOG
            doc_dir = Path(__file__).parent.parent / "doc"
            # Remove a possible 'v' prefix from the version number
            version_clean = remote_version.lstrip('v')
            changelog_file = doc_dir / f"CHANGELOG_v{version_clean}.md"
            
            # Prefer showing the CHANGELOG file
            changelog_shown = False
            if changelog_file.exists():
                try:
                    changelog_content = changelog_file.read_text(encoding='utf-8')
                    print(f"版本 {version_clean} 更新内容:")
                    print("========================================")
                    print(changelog_content)
                    print("========================================")
                    changelog_shown = True
                except Exception as e:
                    print(f"[警告] 无法读取更新文档: {e}")
            
            # Without a CHANGELOG file, show git log
            if not changelog_shown:
                print("最新更新内容 (最近10条):")
                print("----------------------------------------")
                
                try:
                    result = subprocess.run(
                        [git_cmd, 'log', 'HEAD..origin/main', '--oneline', '--decorate', '--no-color', '-10'],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        check=False
                    )
                    if result.returncode == 0 and result.stdout:
                        print(result.stdout.strip())
                    else:
                        print("(无法获取更新日志)")
                except Exception:
                    print("(无法获取更新日志)")
                
                print("----------------------------------------")
            
            return 2  # There is an update
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
