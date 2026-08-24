import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NX_ROOT = REPOSITORY_ROOT / "integrations" / "nx2212"
PROJECT_PATH = NX_ROOT / "BrepMFRNx2212.vcxproj"
DLL_PATH = NX_ROOT / "build" / "Release" / "BrepMFRNx2212.dll"
MSBUILD = {"m": "http://schemas.microsoft.com/developer/msbuild/2003"}


def _project_values(root, name):
    return [
        element.text or ""
        for element in root.findall(".//m:{0}".format(name), MSBUILD)
    ]


def _windows_build_tools():
    if sys.platform != "win32":
        pytest.skip("NX2212 build verification requires Windows")

    nx_dir = os.environ.get("UGII_BASE_DIR")
    if not nx_dir:
        pytest.skip("UGII_BASE_DIR is not configured")
    assert (Path(nx_dir) / "UGOPEN" / "NXOpen" / "UI.hxx").is_file()
    assert (Path(nx_dir) / "UGOPEN" / "libnxopencpp.lib").is_file()

    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    assert program_files_x86
    vswhere = (
        Path(program_files_x86)
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    assert vswhere.is_file()
    discovered = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
    ).stdout.strip()
    assert discovered
    visual_studio = Path(discovered.splitlines()[-1])

    msbuild = (
        visual_studio
        / "MSBuild"
        / "Current"
        / "Bin"
        / "amd64"
        / "MSBuild.exe"
    )
    v142_version_file = (
        visual_studio
        / "VC"
        / "Auxiliary"
        / "Build"
        / "Microsoft.VCToolsVersion.v142.default.txt"
    )
    assert msbuild.is_file()
    assert v142_version_file.is_file()
    v142_version = v142_version_file.read_text(encoding="utf-8-sig").strip()
    dumpbin = (
        visual_studio
        / "VC"
        / "Tools"
        / "MSVC"
        / v142_version
        / "bin"
        / "HostX64"
        / "x64"
        / "dumpbin.exe"
    )
    assert dumpbin.is_file()
    return msbuild, dumpbin


def test_nx2212_project_declares_compatible_portable_contract():
    assert PROJECT_PATH.is_file()
    root = ET.parse(str(PROJECT_PATH)).getroot()
    configurations = {
        node.attrib["Include"]
        for node in root.findall("m:ItemGroup/m:ProjectConfiguration", MSBUILD)
    }
    assert configurations == {"Release|x64"}
    assert _project_values(root, "ConfigurationType") == ["DynamicLibrary"]
    assert _project_values(root, "PlatformToolset") == ["v142"]
    assert _project_values(root, "WindowsTargetPlatformVersion") == ["10.0"]
    assert _project_values(root, "RuntimeLibrary") == ["MultiThreadedDLL"]

    normalized = (
        PROJECT_PATH.read_text(encoding="utf-8-sig")
        .replace("/", "\\")
        .casefold()
    )
    assert "$(ugii_base_dir)\\ugopen" in normalized
    assert "libnxopencpp.lib" in normalized
    assert "libnxopenuicpp.lib" in normalized
    assert "libufun.lib" not in normalized
    assert "nx 10.0" not in normalized
    assert "f:\\program files\\siemens" not in normalized


def test_nx2212_plugin_builds_and_exports_user_functions():
    msbuild, dumpbin = _windows_build_tools()
    completed = subprocess.run(
        [
            str(msbuild),
            str(PROJECT_PATH),
            "/t:Rebuild",
            "/p:Configuration=Release",
            "/p:Platform=x64",
            "/m",
            "/verbosity:minimal",
        ],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert DLL_PATH.is_file()
    assert DLL_PATH.stat().st_size > 0

    exports = subprocess.run(
        [str(dumpbin), "/exports", str(DLL_PATH)],
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
    ).stdout
    assert re.search(r"\bufusr\b", exports)
    assert re.search(r"\bufusr_ask_unload\b", exports)
