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
    parser = argparse.ArgumentParser(description='Version check script')
    parser.add_argument('--brief', action='store_true', help='Brief mode: only show the version and the update hint')
    parser.add_argument('--export-vars', action='store_true', help='Export in environment variable format (for bat scripts)')
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
        print("Manga Translator - starting")
        print("========================================")
        print(f"Current version - {current_version}")
        
        if remote_version == "unknown":
            print("")
            print("[WARNING] Cannot get the remote version information; possibly a network problem")
            print("")
            return 1
        if current_version != remote_version:
            print(f"Remote version - {remote_version}")
            print("")
            print("[NOTICE] A new version is available!")
            print("Run Win-Install-or-Update.bat to update")
            print("")
        else:
            print("")
            print("[INFO] Already up to date")
            print("")
    else:
        # Detailed mode - for script 4 (update and maintenance)
        print(f"Current version - {current_version}")
        print(f"Remote version - {remote_version}")
        
        # Check whether there is an update
        if remote_version == "unknown":
            print("")
            print("[WARNING] Cannot get the remote version information; possibly a network problem")
            return 1
        elif current_version == remote_version:
            print("")
            print("[INFO] Already up to date")
            return 0
        else:
            print("")
            print("[New version found]")
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
                    print(f"Version {version_clean} changes:")
                    print("========================================")
                    print(changelog_content)
                    print("========================================")
                    changelog_shown = True
                except Exception as e:
                    print(f"[WARNING] Cannot read the update notes: {e}")
            
            # Without a CHANGELOG file, show git log
            if not changelog_shown:
                print("Latest changes (last 10):")
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
                        print("(cannot get the change log)")
                except Exception:
                    print("(cannot get the change log)")
                
                print("----------------------------------------")
            
            return 2  # There is an update
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
