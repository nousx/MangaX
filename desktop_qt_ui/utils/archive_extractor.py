"""
Tool for extracting images from archives and documents.
Supports the PDF, EPUB and CBZ formats
"""
import json
import os
import posixpath
import re
import shutil
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from typing import List, Optional, Tuple

from manga_translator.image_formats import SUPPORTED_IMAGE_EXTENSIONS

# Supported archive and document formats
ARCHIVE_EXTENSIONS = {'.pdf', '.epub', '.cbz', '.cbr', '.zip'}

# Supported image formats
IMAGE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS

ORIGINAL_IMAGE_DIRNAME = 'original_images'
ARCHIVE_SOURCE_MARKER_FILENAME = '.archive_source.txt'
EXTRACT_META_FILENAME = '.extract_meta.json'

# ---------------------------------------------------------------------------
# Extraction limits (archive-bomb protection)
#
# Generous for real manga chapters/volumes (hundreds of pages, large PNGs), but
# bounded so that a small crafted archive cannot exhaust RAM or disk space.
# ---------------------------------------------------------------------------
# Maximum number of entries an archive may list at all (directories included).
MAX_ARCHIVE_ENTRY_COUNT = 100_000
# Maximum number of images extracted from one archive/document.
MAX_ARCHIVE_IMAGE_COUNT = 10_000
# Maximum uncompressed size of a single extracted file.
MAX_ARCHIVE_MEMBER_SIZE = 512 * 1024 * 1024  # 512 MiB
# Maximum total uncompressed size extracted from one archive/document.
MAX_ARCHIVE_TOTAL_SIZE = 8 * 1024 * 1024 * 1024  # 8 GiB
# Maximum size of a metadata member read into memory (EPUB OPF / XHTML pages).
MAX_ARCHIVE_METADATA_SIZE = 16 * 1024 * 1024  # 16 MiB
# Members are streamed to disk in chunks of this size.
EXTRACT_CHUNK_SIZE = 1024 * 1024  # 1 MiB


class ArchiveLimitError(ValueError):
    """Raised when an archive exceeds the extraction limits (possible archive bomb)."""


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if size < 1024 or unit == 'GiB':
            return f"{size:.0f} {unit}" if unit == 'B' else f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


class _ExtractionBudget:
    """
    Tracks what one extraction run has written and enforces the limits.

    Limits are read from the module constants when the budget is created.
    Sizes are counted from the bytes actually decompressed, never trusted
    from the archive headers alone.
    """

    def __init__(self, archive_path: str):
        self.archive_name = os.path.basename(archive_path)
        self.max_entries = MAX_ARCHIVE_ENTRY_COUNT
        self.max_images = MAX_ARCHIVE_IMAGE_COUNT
        self.max_member_size = MAX_ARCHIVE_MEMBER_SIZE
        self.max_total_size = MAX_ARCHIVE_TOTAL_SIZE
        self.image_count = 0
        self.total_size = 0
        self.written_paths: List[str] = []

    def _fail(self, reason: str) -> None:
        raise ArchiveLimitError(
            f"Archive '{self.archive_name}' exceeds the extraction limits and was not extracted: {reason}"
        )

    def check_entry_count(self, entry_count: int) -> None:
        if entry_count > self.max_entries:
            self._fail(f"it lists {entry_count} entries (limit {self.max_entries})")

    def check_declared(self, declared_sizes) -> None:
        """Fail fast, before writing anything, using the sizes declared in the headers."""
        sizes = [max(0, int(size or 0)) for size in declared_sizes]
        if len(sizes) > self.max_images:
            self._fail(f"it contains {len(sizes)} images (limit {self.max_images})")
        for size in sizes:
            if size > self.max_member_size:
                self._fail(
                    f"a file is {_format_size(size)} uncompressed (limit {_format_size(self.max_member_size)} per file)"
                )
        total = sum(sizes)
        if total > self.max_total_size:
            self._fail(
                f"total uncompressed size is {_format_size(total)} (limit {_format_size(self.max_total_size)})"
            )

    def add_image(self) -> None:
        self.image_count += 1
        if self.image_count > self.max_images:
            self._fail(f"it contains more than {self.max_images} images")

    def add_bytes(self, member_size_so_far: int, chunk_size: int) -> None:
        if member_size_so_far > self.max_member_size:
            self._fail(
                f"a file is larger than {_format_size(self.max_member_size)} uncompressed"
            )
        self.total_size += chunk_size
        if self.total_size > self.max_total_size:
            self._fail(
                f"total uncompressed size is larger than {_format_size(self.max_total_size)}"
            )

    def copy_stream(self, src, output_path: str) -> None:
        """Stream `src` to `output_path` in chunks, enforcing the limits."""
        self.add_image()
        self.written_paths.append(output_path)
        member_size = 0
        with open(output_path, 'wb') as dst:
            while True:
                chunk = src.read(EXTRACT_CHUNK_SIZE)
                if not chunk:
                    break
                member_size += len(chunk)
                self.add_bytes(member_size, len(chunk))
                dst.write(chunk)

    def write_bytes(self, data: bytes, output_path: str) -> None:
        """Write an in-memory image (e.g. extracted from a PDF), enforcing the limits."""
        self.add_image()
        self.add_bytes(len(data), len(data))
        self.written_paths.append(output_path)
        with open(output_path, 'wb') as dst:
            dst.write(data)

    def register_file(self, output_path: str) -> None:
        """Account for a file written by a third-party API (e.g. a rendered page)."""
        self.written_paths.append(output_path)
        self.add_image()
        try:
            size = os.path.getsize(output_path)
        except OSError:
            size = 0
        self.add_bytes(size, size)

    def remove_partial_output(self) -> None:
        """Delete everything this run wrote (used when extraction is aborted)."""
        for path in self.written_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        self.written_paths.clear()


def _read_member_bounded(zf, name: str, limit: Optional[int] = None) -> bytes:
    """Read a small metadata member into memory, refusing oversized ones."""
    limit = MAX_ARCHIVE_METADATA_SIZE if limit is None else limit
    with zf.open(name) as src:
        data = src.read(limit + 1)
    if len(data) > limit:
        raise ArchiveLimitError(
            f"Archive member '{name}' is larger than {_format_size(limit)} and was not read"
        )
    return data


def is_archive_file(file_path: str) -> bool:
    """Check whether a file is a supported archive or document format"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in ARCHIVE_EXTENSIONS


def get_output_extract_dir(output_base_dir: str, archive_path: str) -> str:
    """Get the extraction folder under the output folder: <output folder>/<file name>/original_images"""
    archive_name = os.path.splitext(os.path.basename(archive_path))[0]
    return os.path.join(output_base_dir, archive_name, ORIGINAL_IMAGE_DIRNAME)

def get_output_extract_root(output_base_dir: str, archive_path: str) -> str:
    """Get the extraction root folder: <output folder>/<file name>"""
    archive_name = os.path.splitext(os.path.basename(archive_path))[0]
    return os.path.join(output_base_dir, archive_name)

def get_output_extract_marker_path(output_base_dir: str, archive_path: str) -> str:
    """Get the path of the file that marks the source archive."""
    return os.path.join(
        get_output_extract_root(output_base_dir, archive_path),
        ARCHIVE_SOURCE_MARKER_FILENAME
    )

def _normalize_abs_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))

def _build_extract_meta(archive_path: str) -> dict:
    return {
        'archive_path': _normalize_abs_path(archive_path),
        'archive_mtime': int(os.path.getmtime(archive_path)) if os.path.exists(archive_path) else 0,
        'archive_size': int(os.path.getsize(archive_path)) if os.path.exists(archive_path) else 0,
    }

def _get_extract_meta_path(output_dir: str) -> str:
    return os.path.join(output_dir, EXTRACT_META_FILENAME)

def _read_extract_meta(output_dir: str) -> Optional[dict]:
    meta_path = _get_extract_meta_path(output_dir)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None

def _write_extract_meta(output_dir: str, archive_path: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    meta_path = _get_extract_meta_path(output_dir)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(_build_extract_meta(archive_path), f, ensure_ascii=False, indent=2)

def _clear_extract_output_dir(output_dir: str) -> None:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

def check_output_extract_conflict(output_base_dir: str, archive_path: str) -> bool:
    """
    Check whether an extraction folder of the same name conflicts with the current archive.
    True means there is a conflict (a folder of the same name whose source is not the current archive_path).
    """
    root_dir = get_output_extract_root(output_base_dir, archive_path)
    if not os.path.isdir(root_dir):
        return False

    marker_path = get_output_extract_marker_path(output_base_dir, archive_path)
    if not os.path.exists(marker_path):
        # For older versions: try to read the metadata of the extraction folder to identify its source
        extract_dir = get_output_extract_dir(output_base_dir, archive_path)
        cached_meta = _read_extract_meta(extract_dir)
        if cached_meta and cached_meta.get('archive_path') == _normalize_abs_path(archive_path):
            return False
        # Without usable metadata it is treated as a conflict, the cautious choice, so a folder with the same name is not reused by mistake
        return True

    try:
        with open(marker_path, 'r', encoding='utf-8') as f:
            recorded_source = f.read().strip()
    except Exception:
        return True

    if not recorded_source:
        return True

    return _normalize_abs_path(recorded_source) != _normalize_abs_path(archive_path)

def clear_output_extract_root(output_base_dir: str, archive_path: str) -> None:
    """Delete the extraction root folder of the same name (for handling a conflict in overwrite mode)."""
    root_dir = get_output_extract_root(output_base_dir, archive_path)
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir, ignore_errors=True)

def write_output_extract_marker(output_base_dir: str, archive_path: str) -> None:
    """Write the source marker of the archive, used to recognise conflicts between folders of the same name."""
    marker_path = get_output_extract_marker_path(output_base_dir, archive_path)
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, 'w', encoding='utf-8') as f:
        f.write(_normalize_abs_path(archive_path))


def get_temp_extract_dir(archive_path: str) -> str:
    """Get the temporary extraction folder of an archive"""
    # Use a fixed subfolder of the system temporary folder, which is easier to manage
    base_temp = os.path.join(tempfile.gettempdir(), 'manga_translator_archives')
    os.makedirs(base_temp, exist_ok=True)
    
    # Build a unique folder name from the file name and the modification time
    archive_name = os.path.splitext(os.path.basename(archive_path))[0]
    mtime = int(os.path.getmtime(archive_path)) if os.path.exists(archive_path) else 0
    unique_name = f"{archive_name}_{mtime}"
    
    return os.path.join(base_temp, unique_name)


def extract_images_from_pdf(pdf_path: str, output_dir: str) -> List[str]:
    """Extract the images from a PDF file (the embedded originals are preferred; a page without one is rendered instead)"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("需要安装 PyMuPDF: pip install PyMuPDF")

    os.makedirs(output_dir, exist_ok=True)
    extracted_images = []
    img_count = 0
    budget = _ExtractionBudget(pdf_path)

    doc = None
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            imgs = page.get_images(full=True)
            if imgs:
                # Extract every embedded image of the page
                for img in imgs:
                    xref = img[0]
                    try:
                        base = doc.extract_image(xref)
                        img_count += 1
                        image_path = os.path.join(output_dir, f"page_{img_count:04d}.{base['ext']}")
                        budget.write_bytes(base['image'], image_path)
                        extracted_images.append(image_path)
                    except ArchiveLimitError:
                        raise
                    except Exception:
                        pass
            else:
                # No embedded image (a text-only or vector page): fall back to rendering it as PNG
                try:
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)
                    img_count += 1
                    image_path = os.path.join(output_dir, f"page_{img_count:04d}.png")
                    pix.save(image_path)
                    pix = None
                    budget.register_file(image_path)
                    extracted_images.append(image_path)
                except ArchiveLimitError:
                    raise
                except Exception:
                    pass
    except ArchiveLimitError:
        budget.remove_partial_output()
        raise
    finally:
        if doc is not None:
            doc.close()

    return sorted(extracted_images)


def extract_images_from_epub(epub_path: str, output_dir: str) -> List[str]:
    """
    Extract the images from an EPUB file in the actual reading order of the book.
    1. Parse the EPUB manifest (<spine> + <manifest>), so the order is strictly the reading order;
    2. Prefer extracting the original high-resolution image each page refers to directly (no loss; the resolution and format of the original are kept);
    3. For a text-only / SVG page, or a page without an image file of its own, fall back to rendering the page as PNG with PyMuPDF (fitz);
    4. Everything is named page_{count:04d}.{ext}, exactly as in the PDF handling;
    5. Errors are handled with a fallback automatically, so no page is ever missed or out of order.
    """
    os.makedirs(output_dir, exist_ok=True)
    extracted_images = []
    img_count = 0
    budget = _ExtractionBudget(epub_path)

    fitz_doc = None
    try:
        import fitz
        try:
            fitz_doc = fitz.open(epub_path)
        except Exception:
            fitz_doc = None
    except ImportError:
        fitz_doc = None

    try:
        with zipfile.ZipFile(epub_path, 'r') as zf:
            namelist = zf.namelist()
            budget.check_entry_count(len(namelist))
            budget.check_declared(
                info.file_size for info in zf.infolist()
                if not info.is_dir() and os.path.splitext(info.filename)[1].lower() in IMAGE_EXTENSIONS
            )
            lower_map = {name.lower(): name for name in namelist}

            # 1. Find the path of the OPF manifest
            opf_path = None
            try:
                container = ET.fromstring(_read_member_bounded(zf, 'META-INF/container.xml'))
                rootfile = container.find('.//{*}rootfile')
                if rootfile is not None and rootfile.get('full-path'):
                    fp = rootfile.get('full-path')
                    opf_path = lower_map.get(fp.lower(), fp)
            except Exception:
                pass

            if not opf_path:
                opf_path = next((name for name in namelist if name.lower().endswith('.opf')), None)

            # 2. Parse the OPF manifest and the reading order
            ordered_targets = []
            if opf_path and opf_path in namelist:
                try:
                    opf_dir = posixpath.dirname(opf_path)
                    opf = ET.fromstring(_read_member_bounded(zf, opf_path))
                    manifest = {
                        item.get('id'): item.get('href', '')
                        for item in opf.findall('.//{*}item')
                        if item.get('id')
                    }
                    spine_ids = [ref.get('idref') for ref in opf.findall('.//{*}itemref') if ref.get('idref')]

                    img_pattern = re.compile(
                        r'(?:src|href)=["\']([^"\']+\.(?:' + '|'.join(e.lstrip('.') for e in IMAGE_EXTENSIONS) + r'))',
                        re.IGNORECASE
                    )

                    seen_in_spine = set()
                    for sid in spine_ids:
                        href = manifest.get(sid)
                        if not href:
                            continue
                        target = posixpath.normpath(posixpath.join(opf_dir, urllib.parse.unquote(href)))
                        ext = os.path.splitext(target)[1].lower()

                        if ext in IMAGE_EXTENSIONS:
                            real_p = lower_map.get(target.lower())
                            if real_p and real_p not in seen_in_spine:
                                seen_in_spine.add(real_p)
                                ordered_targets.append((real_p, None))
                        else:
                            real_html = lower_map.get(target.lower())
                            found = False
                            if real_html:
                                html_text = _read_member_bounded(zf, real_html).decode('utf-8', errors='replace')
                                for match in img_pattern.findall(html_text):
                                    img_ref = urllib.parse.unquote(match.split('?')[0].split('#')[0])
                                    img_full = posixpath.normpath(posixpath.join(posixpath.dirname(real_html), img_ref))
                                    real_img = lower_map.get(img_full.lower())
                                    if real_img and real_img not in seen_in_spine:
                                        seen_in_spine.add(real_img)
                                        ordered_targets.append((real_img, None))
                                        found = True
                            if not found:
                                # A text-only / SVG page, or a page without an image file of its own: record its page number in the spine for rendering with fitz
                                ordered_targets.append((None, len(ordered_targets)))
                except Exception:
                    ordered_targets.clear()

            # 3. Extract the original high-resolution images in reading order (or fall back to rendering)
            extracted_paths = set()
            for zip_rel, spine_page_idx in ordered_targets:
                if zip_rel:
                    ext = os.path.splitext(zip_rel)[1].lower() or '.png'
                    img_count += 1
                    out_path = os.path.join(output_dir, f"page_{img_count:04d}{ext}")
                    with zf.open(zip_rel) as src:
                        budget.copy_stream(src, out_path)
                    extracted_images.append(out_path)
                    extracted_paths.add(zip_rel)
                elif fitz_doc is not None and spine_page_idx is not None and spine_page_idx < len(fitz_doc):
                    try:
                        mat = fitz.Matrix(2.0, 2.0)
                        pix = fitz_doc[spine_page_idx].get_pixmap(matrix=mat)
                        img_count += 1
                        out_path = os.path.join(output_dir, f"page_{img_count:04d}.png")
                        pix.save(out_path)
                        budget.register_file(out_path)
                        extracted_images.append(out_path)
                    except ArchiveLimitError:
                        raise
                    except Exception:
                        pass

            # 4. Append the remaining images that the spine does not reference explicitly (so that no page is ever missed)
            remaining = [
                name for name in namelist
                if name not in extracted_paths and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
            ]
            remaining.sort(key=natural_sort_key)
            for rem in remaining:
                ext = os.path.splitext(rem)[1].lower() or '.png'
                img_count += 1
                out_path = os.path.join(output_dir, f"page_{img_count:04d}{ext}")
                with zf.open(rem) as src:
                    budget.copy_stream(src, out_path)
                extracted_images.append(out_path)

            # 5. When no image was extracted at all, render the whole book with fitz as a last resort
            if not extracted_images and fitz_doc is not None:
                for fitz_page in fitz_doc:
                    try:
                        mat = fitz.Matrix(2.0, 2.0)
                        pix = fitz_page.get_pixmap(matrix=mat)
                        img_count += 1
                        out_path = os.path.join(output_dir, f"page_{img_count:04d}.png")
                        pix.save(out_path)
                        budget.register_file(out_path)
                        extracted_images.append(out_path)
                    except ArchiveLimitError:
                        raise
                    except Exception:
                        pass
    except ArchiveLimitError:
        budget.remove_partial_output()
        raise
    finally:
        if fitz_doc is not None:
            fitz_doc.close()

    return sorted(extracted_images)


def _extract_image_members(archive, archive_path: str, output_dir: str) -> List[str]:
    """
    Extract the image members of an opened ZIP/RAR archive into `output_dir`.

    Members are streamed to disk in chunks and checked against the extraction
    limits.  If a limit is exceeded, everything written by this call is removed
    and ArchiveLimitError is raised.
    """
    budget = _ExtractionBudget(archive_path)
    extracted_images = []

    all_entries = archive.infolist()
    budget.check_entry_count(len(all_entries))

    # Collect all image files and sort them
    image_files = []
    for file_info in all_entries:
        if file_info.is_dir():
            continue
        ext = os.path.splitext(file_info.filename)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            image_files.append(file_info)

    # Fail fast on the declared sizes before anything is written.
    budget.check_declared(file_info.file_size for file_info in image_files)

    # Natural sort by file name
    image_files.sort(key=lambda x: natural_sort_key(x.filename))

    try:
        for idx, file_info in enumerate(image_files):
            # basename only: member paths can never escape output_dir
            base_name = os.path.basename(file_info.filename.replace('\\', '/'))
            # Add a number prefix to keep the order
            new_name = f"{idx:04d}_{base_name}"
            output_path = os.path.join(output_dir, new_name)

            with archive.open(file_info) as src:
                budget.copy_stream(src, output_path)
            extracted_images.append(output_path)
    except ArchiveLimitError:
        budget.remove_partial_output()
        raise

    return extracted_images


def extract_images_from_cbz(cbz_path: str, output_dir: str) -> List[str]:
    """Extract the images from a CBZ (Comic Book ZIP) file"""
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(cbz_path, 'r') as zf:
        return _extract_image_members(zf, cbz_path, output_dir)


def extract_images_from_cbr(cbr_path: str, output_dir: str) -> List[str]:
    """Extract the images from a CBR (Comic Book RAR) file"""
    try:
        import rarfile
    except ImportError:
        raise ImportError("需要安装 rarfile: pip install rarfile")

    os.makedirs(output_dir, exist_ok=True)

    with rarfile.RarFile(cbr_path, 'r') as rf:
        return _extract_image_members(rf, cbr_path, output_dir)


def natural_sort_key(s: str):
    """Natural sort key; numbers sort as numbers"""
    import re
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]


def extract_images_from_archive(archive_path: str, output_dir: Optional[str] = None) -> Tuple[List[str], str]:
    """
    Extract the images from an archive or a document

    Args:
        archive_path: path of the archive or document
        output_dir: output folder; a temporary folder is used when None

    Returns:
        (list of the extracted image paths, output folder)
    """
    if output_dir is None:
        output_dir = get_temp_extract_dir(archive_path)
    
    expected_meta = _build_extract_meta(archive_path)

    # When the folder exists and the cache metadata matches, return the cached result directly
    if os.path.exists(output_dir):
        existing_images = []
        for f in os.listdir(output_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                existing_images.append(os.path.join(output_dir, f))
        cached_meta = _read_extract_meta(output_dir)
        if existing_images and cached_meta == expected_meta:
            return sorted(existing_images), output_dir
        # The folder exists but the cache cannot be used (a source or version mismatch, or leftover data): empty it and extract again
        _clear_extract_output_dir(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    ext = os.path.splitext(archive_path)[1].lower()
    
    try:
        if ext == '.pdf':
            images = extract_images_from_pdf(archive_path, output_dir)
        elif ext == '.epub':
            images = extract_images_from_epub(archive_path, output_dir)
        elif ext in {'.cbz', '.zip'}:
            images = extract_images_from_cbz(archive_path, output_dir)
        elif ext == '.cbr':
            images = extract_images_from_cbr(archive_path, output_dir)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    except ArchiveLimitError:
        # Extraction was aborted: drop the (already emptied) output directory so
        # that no partial result can be mistaken for a valid cache later.
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    _write_extract_meta(output_dir, archive_path)
    return images, output_dir


def cleanup_temp_archives():
    """Remove all temporary extraction folders"""
    base_temp = os.path.join(tempfile.gettempdir(), 'manga_translator_archives')
    if os.path.exists(base_temp):
        shutil.rmtree(base_temp, ignore_errors=True)


def cleanup_archive_temp(archive_path: str):
    """Remove the temporary extraction folder of the given archive"""
    temp_dir = get_temp_extract_dir(archive_path)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
