"""
Photoshop PSD export module.
Uses ExtendScript (.jsx) to produce an editable PSD file
"""

import logging
import os
import platform
import subprocess
import tempfile

# import math
from typing import Optional

from . import Context
from ..rendering.rich_text import plain_text_of
from manga_translator.utils.swallowed import note_ignored_error

logger = logging.getLogger(__name__)


def psd_export_requested(config) -> bool:
    """True when the settings ask for a PSD file or for its Photoshop script alone.

    'Script only' is a complete request by itself: it must not also need the
    'export editable PSD' switch, which is what starts Photoshop.
    """
    cli = getattr(config, 'cli', None)
    return bool(getattr(cli, 'export_editable_psd', False) or getattr(cli, 'psd_script_only', False))


def resolve_photoshop_font(config) -> str | None:
    """Get the render font used by the PSD text layers."""
    render_cfg = getattr(config, 'render', None)
    font_family = getattr(render_cfg, 'font_family', None)
    if isinstance(font_family, str) and font_family.strip():
        return font_family.strip()
    return None


def _translation_plain_text(value) -> str:
    # Thin wrapper: the one rich-text to plain-text implementation is rendering.rich_text.plain_text_of
    return plain_text_of(value)


# Alignment mapped to the Justification enum of Photoshop
ALIGNMENT_TO_PS_JUSTIFICATION = {
    "left": "Justification.LEFT",
    "right": "Justification.RIGHT",
    "center": "Justification.CENTER",
}

# Text direction mapping
DIRECTION_TO_PS_DIRECTION = {
    "h": "Direction.HORIZONTAL",
    "v": "Direction.VERTICAL",
    "hr": "Direction.HORIZONTAL",  # Right to left needs extra handling
    "vr": "Direction.VERTICAL",
}


# JSX script template
JSX_TEMPLATE = """
#target photoshop

// 设置单位为像素
app.preferences.rulerUnits = Units.PIXELS;
app.preferences.typeUnits = TypeUnits.PIXELS;

// 定义错误日志文件路径
var ERROR_FILE_PATH = '{error_file}';

// 字体查找函数：根据字体名称查找 PostScript 名称
// Photoshop 的 textItem.font 需要使用 PostScript 名称
function findFontPostScriptName(fontName) {{
    if (!fontName) return null;
    
    var lowerName = fontName.toLowerCase();
    var fonts = app.fonts;
    
    // 第一轮：精确匹配 PostScript 名称（不区分大小写）
    for (var i = 0; i < fonts.length; i++) {{
        if (fonts[i].postScriptName.toLowerCase() === lowerName) {{
            return fonts[i].postScriptName;
        }}
    }}
    
    // 第二轮：精确匹配字体名称（不区分大小写）
    for (var i = 0; i < fonts.length; i++) {{
        if (fonts[i].name.toLowerCase() === lowerName) {{
            return fonts[i].postScriptName;
        }}
    }}
    
    // 第三轮：精确匹配字体家族名称（不区分大小写）
    for (var i = 0; i < fonts.length; i++) {{
        if (fonts[i].family.toLowerCase() === lowerName) {{
            return fonts[i].postScriptName;
        }}
    }}
    
    // 第四轮：部分匹配（字体名称包含搜索词）
    for (var i = 0; i < fonts.length; i++) {{
        var psName = fonts[i].postScriptName.toLowerCase();
        var name = fonts[i].name.toLowerCase();
        var family = fonts[i].family.toLowerCase();
        
        if (psName.indexOf(lowerName) !== -1 || 
            name.indexOf(lowerName) !== -1 || 
            family.indexOf(lowerName) !== -1) {{
            return fonts[i].postScriptName;
        }}
    }}
    
    // 第五轮：搜索词包含在字体名称中（反向匹配）
    for (var i = 0; i < fonts.length; i++) {{
        var psName = fonts[i].postScriptName.toLowerCase();
        var name = fonts[i].name.toLowerCase();
        var family = fonts[i].family.toLowerCase();
        
        if (lowerName.indexOf(psName) !== -1 || 
            lowerName.indexOf(name) !== -1 || 
            lowerName.indexOf(family) !== -1) {{
            return fonts[i].postScriptName;
        }}
    }}
    
    return null;
}}

// 縦中横（Tate-chu-yoko）处理函数
// 通过设置 baselineDirection 为 "Crs " (Cross) 实现
function applyTateChuYoko(textLayer, charStart, charEnd, fontSize) {{
    try {{
        app.activeDocument.activeLayer = textLayer;
        
        var idsetd = charIDToTypeID("setd");
        var desc1 = new ActionDescriptor();
        var idnull = charIDToTypeID("null");
        var ref1 = new ActionReference();
        ref1.putEnumerated(charIDToTypeID("TxLr"), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));
        desc1.putReference(idnull, ref1);
        
        var idT = charIDToTypeID("T   ");
        var desc2 = new ActionDescriptor();
        var idTxtt = charIDToTypeID("Txtt");
        var list1 = new ActionList();
        var desc3 = new ActionDescriptor();
        
        desc3.putInteger(charIDToTypeID("From"), charStart);
        desc3.putInteger(charIDToTypeID("T   "), charEnd);
        
        var idTxtS = charIDToTypeID("TxtS");
        var desc4 = new ActionDescriptor();
        desc4.putBoolean(stringIDToTypeID("styleSheetHasParent"), true);
        
        // 保留字体大小
        var idPxl = charIDToTypeID("#Pxl");
        desc4.putUnitDouble(charIDToTypeID("Sz  "), idPxl, fontSize);
        desc4.putUnitDouble(stringIDToTypeID("impliedFontSize"), idPxl, fontSize);
        
        // 关键：设置 baselineDirection 为 "Crs " (Cross) 实现縦中横
        var idbaselineDirection = stringIDToTypeID("baselineDirection");
        desc4.putEnumerated(idbaselineDirection, idbaselineDirection, charIDToTypeID("Crs "));
        
        desc3.putObject(idTxtS, idTxtS, desc4);
        list1.putObject(idTxtt, desc3);
        desc2.putList(idTxtt, list1);
        desc1.putObject(idT, charIDToTypeID("TxLr"), desc2);
        
        executeAction(idsetd, desc1, DialogModes.NO);
        $.writeln('Applied TateChuYoko at position ' + charStart + '-' + charEnd);
        return true;
    }} catch (e) {{
        $.writeln('WARNING: Failed to apply TateChuYoko: ' + e.message);
        return false;
    }}
}}

// 查找文本中需要縦中横的字符位置
function findTateChuYokoPositions(text) {{
    var positions = [];
    var tcyChars = ['⁉', '⁈', '‼', '⁇'];
    
    for (var i = 0; i < text.length; i++) {{
        var ch = text.charAt(i);
        for (var j = 0; j < tcyChars.length; j++) {{
            if (ch === tcyChars[j]) {{
                positions.push([i, i + 1]);
                break;
            }}
        }}
    }}
    return positions;
}}

try {{
    // 打开原始图片
    var inputFile = new File('{input_file}');
    
    // 检查文件是否存在
    if (!inputFile.exists) {{
        $.writeln('ERROR: Input file does not exist: ' + inputFile.fsName);
        throw new Error('Input file does not exist: ' + inputFile.fsName);
    }}
    
    $.writeln('Opening input file: ' + inputFile.fsName);
    var doc = app.open(inputFile);
    $.writeln('Document opened successfully');
    
    // 设置文档颜色模式为 RGB（如果不是的话）
    try {{
        if (doc.mode != DocumentMode.RGB) {{
            $.writeln('Converting to RGB mode');
            doc.changeMode(ChangeMode.RGB);
        }}
    }} catch (modeError) {{
        $.writeln('WARNING: Could not convert to RGB mode: ' + modeError.message);
    }}
    
    // 重命名背景层或第一个图层为"原图"
    // Photoshop 2020+ 打开纯色PNG时可能不创建背景层，而是普通图层
    var originalLayer = null;
    
    try {{
        // 尝试获取背景层
        if (doc.backgroundLayer) {{
            originalLayer = doc.backgroundLayer;
            $.writeln('Found background layer');
        }}
    }} catch (e) {{
        $.writeln('No background layer found: ' + e.message);
    }}
    
    // 如果没有背景层，使用第一个图层（最底层）
    if (!originalLayer && doc.layers.length > 0) {{
        originalLayer = doc.layers[doc.layers.length - 1];
        $.writeln('Using bottom layer as original: ' + originalLayer.name);
    }}
    
    // 重命名并锁定原图层
    if (originalLayer) {{
        originalLayer.name = '原图 (original)';
        originalLayer.allLocked = true;
        $.writeln('Original layer renamed and locked');
    }} else {{
        $.writeln('WARNING: No original layer found');
    }}
    
    {inpainted_layer_code}
    
    {mask_layer_code}
    
    {text_layers_code}
    
    // 保存为 PSD
    var psdFile = new File('{output_file}');
    $.writeln('Saving PSD to: ' + psdFile.fsName);
    var psdOptions = new PhotoshopSaveOptions();
    psdOptions.embedColorProfile = true;
    psdOptions.alphaChannels = true;
    psdOptions.layers = true;
    psdOptions.spotColors = true;
    
    doc.saveAs(psdFile, psdOptions, true);
    $.writeln('PSD saved successfully');
    doc.close(SaveOptions.DONOTSAVECHANGES);
    $.writeln('Document closed');
    
}} catch (e) {{
    var errorMsg = 'ERROR: ' + e.message + '\\nLine: ' + e.line + '\\nFile: ' + (e.fileName || 'unknown');
    $.writeln(errorMsg);
    
    // 写入错误文件
    try {{
        var errFile = new File(ERROR_FILE_PATH);
        errFile.open('w');
        errFile.write(errorMsg);
        errFile.close();
    }} catch (writeErr) {{
        $.writeln('Failed to write error file: ' + writeErr.message);
    }}
}}
"""

# JSX code template of a single text layer
# In the order of BallonsTranslator: type -> contents -> position -> width/height -> size
# Point text (PointText) is used, which has no bounding box limit
TEXT_LAYER_TEMPLATE = """
    // 文本层 {index}: {name}
    var textLayer{index} = doc.artLayers.add();
    textLayer{index}.kind = LayerKind.TEXT;
    textLayer{index}.name = '{name}';
    
    var textItem{index} = textLayer{index}.textItem;
    // 点文本：不设置 kind，默认就是 POINTTEXT
    
    {font_setup_code}

    // 根据文档分辨率调整尺寸值（Photoshop 内部使用 72 DPI 作为基准）
    // 设置属性顺序：方向 -> 对齐 -> 位置 -> 内容 -> 大小
    var dpiScale{index} = 72 / doc.resolution;
    
    textItem{index}.direction = {direction};
    textItem{index}.justification = {justification};
    textItem{index}.position = [{x}, {y}];
    textItem{index}.contents = '{text}';
    textItem{index}.size = new UnitValue({font_size} * dpiScale{index}, 'pt');
    
    // 设置文字颜色
    var textColor{index} = new SolidColor();
    textColor{index}.rgb.red = {color_r};
    textColor{index}.rgb.green = {color_g};
    textColor{index}.rgb.blue = {color_b};
    textItem{index}.color = textColor{index};
    
    {tracking_code}
    {leading_code}
    {rotation_code}
    {tcy_code}
"""

# Code template of the inpainted layer
INPAINTED_LAYER_TEMPLATE = """
    // 添加修复后的图层
    $.writeln('Adding inpainted layer from: {inpainted_file}');
    var inpaintedFile = new File('{inpainted_file}');
    if (!inpaintedFile.exists) {{
        $.writeln('WARNING: Inpainted file does not exist');
    }} else {{
        var inpaintedDoc = app.open(inpaintedFile);
        inpaintedDoc.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
        inpaintedDoc.close(SaveOptions.DONOTSAVECHANGES);
        doc.activeLayer.name = '修复图 (inpainted)';
        $.writeln('Inpainted layer added successfully');
    }}
"""

# Code template of the mask layer
MASK_LAYER_TEMPLATE = """
    // 添加遮罩层
    $.writeln('Adding mask layer from: {mask_file}');
    var maskFile = new File('{mask_file}');
    if (!maskFile.exists) {{
        $.writeln('WARNING: Mask file does not exist');
    }} else {{
        var maskDoc = app.open(maskFile);
        maskDoc.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
        maskDoc.close(SaveOptions.DONOTSAVECHANGES);
        doc.activeLayer.name = '遮罩 (mask)';
        $.writeln('Mask layer added successfully');
    }}
"""


def escape_jsx_string(text: str) -> str:
    """Escape the special characters of a JSX string (for strings wrapped in single quotes)"""
    if not text:
        return ""
    # Backslashes must be handled first, before the other escapes
    text = text.replace("\\", "\\\\")   # Backslash (when the text has one)
    text = text.replace("'", "\\'")    # Single quote
    text = text.replace("\n", "\\r")   # Line feed (Photoshop uses \r)
    text = text.replace("\r", "\\r")   # Carriage return
    text = text.replace("\t", "    ")  # Tab to space
    text = text.replace("\t", "    ")  # Tab to space
    
    # Replace [BR] and the whitespace around it with a regular expression, ignoring case
    # Both the half-width [BR] and the full-width 【BR】 are handled
    import re
    text = re.sub(r'\s*(?:\[|【)BR(?:\]|】)\s*', '\\r', text, flags=re.IGNORECASE)

    # Final step: handle every possible vertical whitespace character
    # That is \n, \r, \u2028 (Line Separator), \u2029 (Paragraph Separator), \v (Vertical Tab), \f (Form Feed)
    # This turns every physical line break into an escaped \r
    text = re.sub(r'[\r\n\u2028\u2029\v\f]+', '\\r', text)
    
    return text


def escape_jsx_path(path: str) -> str:
    """Turn a file path into a form that is safe to embed in a single-quoted JSX string.

    Photoshop's ExtendScript wraps paths in single quotes; an ASCII single quote in the path
    has to be escaped, otherwise it ends the string early and causes a script syntax error. Windows backslashes
    are turned into forward slashes at the same time, so JSX does not parse them as escape sequences.
    """
    if not path:
        return ""

    # Normalise the path separators first, then escape the special characters of a JSX string.
    return path.replace("\\", "/").replace("'", "\\'")


# Symbols that need tate-chu-yoko (shown horizontally) in vertical text
# Replace multi-character symbols with a single full-width character, so they are not split apart in vertical text
VERTICAL_HORIZONTAL_MAP = {
    # Combined punctuation -> a single full-width character
    "!?": "⁉",      # Exclamation mark + question mark
    "?!": "⁈",      # Question mark + exclamation mark
    "!!": "‼",      # Double exclamation mark
    "??": "⁇",      # Double question mark
    # Half-width -> full-width (looks better in vertical text)
    "!": "！",
    "?": "？",
}


def preprocess_vertical_text(text: str, is_vertical: bool) -> str:
    """
    Preprocess vertical text, handling tate-chu-yoko (horizontal text set inside a vertical line)

    Args:
        text: the original text
        is_vertical: whether the text is vertical

    Returns:
        The processed text
    """
    if not is_vertical or not text:
        return text
    
    result = text
    
    # Full-width symbol combinations (Chinese translations often use full-width)
    result = result.replace("！？", "⁉")
    result = result.replace("？！", "⁈")
    result = result.replace("！！", "‼")
    result = result.replace("？？", "⁇")
    
    # Half-width symbol combinations
    result = result.replace("!?", "⁉")
    result = result.replace("?!", "⁈")
    result = result.replace("!!", "‼")
    result = result.replace("??", "⁇")
    
    # Single half-width -> full-width
    result = result.replace("!", "！")
    result = result.replace("?", "？")
    
    return result


def generate_text_layer_jsx(index: int, text_region, default_font: str, line_spacing: float = None) -> str:
    """Build the JSX code of a single text layer"""
    
    # Text direction (decided early, for the text preprocessing)
    direction = text_region.direction
    is_vertical = direction.startswith('v') if direction else False
    
    # Text content (the translated text)
    # Preprocess for vertical text first (tate-chu-yoko and so on)
    raw_text = _translation_plain_text(text_region.translation)
    processed_text = preprocess_vertical_text(raw_text, is_vertical)
    text = escape_jsx_string(processed_text)
    
    # Second safeguard: remove every possible physical line break, to prevent script syntax errors
    if '\n' in text or '\r' in text:
        # If escape_jsx_string did not clean everything (which should not happen), it is forced here
        logger.warning(f"Text layer {index} still contains physical line breaks, removing them")
        text = text.replace('\n', '\\r').replace('\r', '\\r')
    
    if not text:
        logger.warning(f"Text layer {index} has an empty translation, skipping")
        return ""
    
    logger.debug(f"Text layer {index}: source='{' '.join(text_region.text)[:30]}', translation='{raw_text[:30]}'")
    
    # Position and size - from the dst_points computed at the rendering stage
    # dst_points is an array of shape (1, 4, 2) holding the 4 corners
    pts = text_region.dst_points.reshape(-1, 2)
    x_min = float(pts[:, 0].min())
    y_min = float(pts[:, 1].min())
    x_max = float(pts[:, 0].max())
    y_max = float(pts[:, 1].max())
    w = x_max - x_min
    h = y_max - y_min
    
    # Text direction (Photoshop enum value)
    direction_ps = DIRECTION_TO_PS_DIRECTION.get(direction, "Direction.HORIZONTAL")
    
    # Number of lines (the same regular expression logic as escape_jsx_string)
    import re
    # Both the half-width [BR] and the full-width 【BR】 are handled
    num_lines = len(re.split(r'\s*(?:\[|【)BR(?:\]|】)\s*', raw_text, flags=re.IGNORECASE))
    
    # Font size
    font_size = text_region.font_size
    
    # Line spacing factor (leading factor)
    # Base spacing 0.2 for vertical text, 0.01 for horizontal text
    base_spacing = 0.2 if is_vertical else 0.01
    multiplier = line_spacing if line_spacing is not None else 1.0
    leading_factor = 1.0 + base_spacing * multiplier
    
    # position of the point text
    # With centre alignment, position should be the baseline centre of the first line (horizontal) or first column (vertical)
    # It has to be corrected by the number of lines, so the whole text block is centred
    
    box_center_x = x_min + w / 2
    box_center_y = y_min + h / 2
    
    # Approximate leading in pixels (px)
    # font_size itself is in pixels (px), because text_region.font_size comes from image analysis
    line_height_px = font_size * leading_factor
    
    if is_vertical:
        # Vertical: top aligned (Justification.LEFT usually means Start/Top)
        # y is the top
        y = y_min
        
        # x is the centre of the first column
        # Columns run from right to left, so the first column is the rightmost one
        # To centre the whole block horizontally:
        offset_x = (num_lines - 1) * line_height_px / 2
        x = box_center_x + offset_x
        
        justification = "Justification.LEFT"
        
    else:
        # Horizontal: centre aligned (Justification.CENTER)
        # x stays at the centre
        x = box_center_x
        
        # y is the baseline of the first line
        baseline_offset = font_size * 0.35
        offset_y = (num_lines - 1) * line_height_px / 2
        y = box_center_y - offset_y + baseline_offset
        
        justification = "Justification.CENTER"
    
    # Colour
    color_r, color_g, color_b = text_region.fg_colors[:3] if len(text_region.fg_colors) >= 3 else (0, 0, 0)
    
    # Letter spacing (tracking) - not set, the Photoshop default is used
    tracking_code = ""
    
    # Line spacing (leading)
    # Make sure the value is a float
    leading_val_px = float(font_size) * float(leading_factor)
    # toFixed(2) makes sure it is a number in JS
    leading_code = f"""
    var leadingVal{index} = {leading_val_px} * dpiScale{index};
    textItem{index}.useAutoLeading = false;
    textItem{index}.leading = new UnitValue(leadingVal{index}, 'pt');
    """
    
    # Rotation
    rotation_code = ""
    if abs(text_region.angle) > 1:  # Rotate only when the angle is larger than 1 degree
        # Use the original angle value directly
        rotation_code = f"textLayer{index}.rotate({text_region.angle}, AnchorPosition.MIDDLECENTER);"
    
    # Tate-chu-yoko handling (vertical text only)
    tcy_code = ""
    if is_vertical:
        tcy_chars = ['⁉', '⁈', '‼', '⁇']
        has_tcy = any(c in processed_text for c in tcy_chars)
        if has_tcy:
            tcy_code = f"""
    // 应用 Photoshop 縦中横
    var tcyPositions{index} = findTateChuYokoPositions(textItem{index}.contents);
    var tcyFontSize{index} = {font_size};
    for (var ti{index} = 0; ti{index} < tcyPositions{index}.length; ti{index}++) {{
        applyTateChuYoko(textLayer{index}, tcyPositions{index}[ti{index}][0], tcyPositions{index}[ti{index}][1], tcyFontSize{index});
    }}
"""
    
    # Font setting code
    if default_font:
        font_setup_code = f"""// 查找字体的 PostScript 名称
    var requestedFont{index} = '{default_font}';
    var fontPS{index} = findFontPostScriptName(requestedFont{index});
    
    if (!fontPS{index}) {{
        var msg = 'Font not found: "' + requestedFont{index} + '". Please install this font or use a different font.';
        $.writeln('ERROR: ' + msg);
        var errFile = new File(ERROR_FILE_PATH);
        errFile.open('w');
        errFile.write(msg);
        errFile.close();
        throw new Error(msg);
    }}
    
    $.writeln('Font mapping: "' + requestedFont{index} + '" -> "' + fontPS{index} + '"');
    
    // 设置字体（使用 PostScript 名称）
    try {{
        textItem{index}.font = fontPS{index};
    }} catch (e) {{
        var msg = 'Failed to set font "' + fontPS{index} + '": ' + e.message;
        $.writeln('ERROR: ' + msg);
        var errFile = new File(ERROR_FILE_PATH);
        errFile.open('w');
        errFile.write(msg);
        errFile.close();
        throw new Error(msg);
    }}
"""
    else:
        font_setup_code = f"""// 使用 Photoshop 默认字体
    $.writeln('Using Photoshop default font for layer {index}');
"""
    
    # Text layer name: the translation, with every line break removed so the JSX syntax is not broken
    # 1. Remove [BR] and the full-width 【BR】 (replaced with an empty string)
    safe_name = re.sub(r'\s*(?:\[|【)BR(?:\]|】)\s*', '', text_region.translation, flags=re.IGNORECASE)
    # 2. Remove every physical line break (replaced with an empty string)
    safe_name = re.sub(r'[\r\n\u2028\u2029\v\f]+', '', safe_name)
    # 3. Escape the special characters and truncate
    name = escape_jsx_string(safe_name[:50])

    return TEXT_LAYER_TEMPLATE.format(
        index=index,
        name=name,
        text=text,
        x=x,
        y=y,
        w=w,
        h=h,
        font_size=font_size,
        color_r=color_r,
        color_g=color_g,
        color_b=color_b,
        justification=justification,
        direction=direction_ps,
        is_vertical='true' if is_vertical else 'false',
        tracking_code=tracking_code,
        leading_code=leading_code,
        rotation_code=rotation_code,
        tcy_code=tcy_code,
        font_setup_code=font_setup_code,
    )


def get_psd_output_path(image_path: str) -> str:
    """
    Get the output path of the PSD file

    Creates the manga_translator_work/psd/ folder next to the original image

    Args:
        image_path: path of the original image

    Returns:
        The full path of the PSD file
    """
    # Folder and file name of the original image
    image_dir = os.path.dirname(os.path.abspath(image_path))
    image_name = os.path.basename(image_path)
    base_name, _ = os.path.splitext(image_name)
    
    # Create the manga_translator_work/psd folder
    psd_dir = os.path.join(image_dir, 'manga_translator_work', 'psd')
    os.makedirs(psd_dir, exist_ok=True)
    
    # Build the PSD file path
    psd_path = os.path.join(psd_dir, f"{base_name}.psd")
    
    return psd_path


def _save_image_like_to_temp(image_data, target_path: str) -> bool:
    """Save the image data of the current session as temporary files, for the PSD export to reuse."""
    try:
        import numpy as np
        from PIL import Image

        if image_data is None:
            return False

        if isinstance(image_data, Image.Image):
            image_to_save = image_data.copy()
        elif isinstance(image_data, np.ndarray):
            image_to_save = Image.fromarray(image_data)
        else:
            logger.warning(f"Unsupported PSD image type: {type(image_data)}")
            return False

        if image_to_save.mode == 'CMYK':
            image_to_save = image_to_save.convert('RGB')

        image_to_save.save(target_path)
        return True
    except Exception as e:
        logger.warning(f"Failed to save temporary PSD image: {e}")
        return False


def photoshop_export(output_file: str, ctx: Context, default_font: str = None, image_path: str = None, verbose: bool = False, result_path_fn=None, line_spacing: float = None, script_only: bool = False):
    """
    Export a PSD file with Photoshop

    Layer structure (bottom to top):
    1. Original image (original) - editor_base when available, otherwise the original image; locked
    2. Inpainted image (inpainted) - the one of the current session when available, otherwise from the work folder
    3. Mask (mask) - when there is one
    4. Text layers - editable

    Args:
        output_file: path of the output PSD file
        ctx: the translation context, with the image and the text region information
        default_font: default font name; when None the Photoshop default font is used
        image_path: path of the original image (used to find editor_base and the inpainted image in the work folder)
        verbose: whether debug mode is on (saves the JSX script to the result folder)
        result_path_fn: function that builds result paths (used to save the debug script)
        line_spacing: line spacing factor
        script_only: when True, only the JSX script is produced and Photoshop is not run
    """
    
    # When default_font is a file path, take the font name from it
    if default_font and (os.path.sep in default_font or '/' in default_font or default_font.endswith('.ttf') or default_font.endswith('.otf')):
        # Take the file name (without extension) from the path as the font name
        font_basename = os.path.splitext(os.path.basename(default_font))[0]
        logger.warning(f"Detected a file path in default_font: {default_font}")
        logger.warning(f"Extracted font name: {font_basename}")
        logger.warning("Tip: Select a font name from the system font list instead of using a font file path")
        default_font = font_basename
    
    # Create temporary files (only for the inpainted image and the mask)
    temp_dir = tempfile.gettempdir()
    inpainted_file = os.path.join(temp_dir, ".ps_inpainted.png")
    mask_file = os.path.join(temp_dir, ".ps_mask.png")
    jsx_file = os.path.join(temp_dir, ".ps_script.jsx")
    error_file = os.path.join(temp_dir, ".ps_error.txt")
    
    # Remove old error files
    if os.path.exists(error_file):
        try:
            os.unlink(error_file)
        except Exception as ignored_error:
            note_ignored_error(ignored_error, "manga_translator/utils/photoshop_export.py:photoshop_export")
            pass
    
    try:
        # The PSD base image prefers editor_base, to match the "original layer" in the editor
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f'Original image path is invalid or the file does not exist: {image_path}')

        from .path_manager import find_work_image_path, get_inpainted_path

        work_image_path = find_work_image_path(image_path)
        if work_image_path and os.path.exists(work_image_path):
            input_file = work_image_path
            logger.info(f"Using editor_base as the PSD base image: {input_file}")
        else:
            input_file = image_path
            logger.info(f"Falling back to the original image as the PSD base image: {input_file}")

        # The inpainted image prefers the result of the current session, so an old image on disk is not picked up
        inpainted_layer_code = ""
        if hasattr(ctx, 'img_inpainted') and ctx.img_inpainted is not None and _save_image_like_to_temp(ctx.img_inpainted, inpainted_file):
            inpainted_layer_code = INPAINTED_LAYER_TEMPLATE.format(
                inpainted_file=escape_jsx_path(inpainted_file)
            )
            logger.info("Using the current session result as the PSD inpainted image")
        elif image_path:
            inpainted_path = get_inpainted_path(image_path, create_dir=False)
            if os.path.exists(inpainted_path):
                inpainted_layer_code = INPAINTED_LAYER_TEMPLATE.format(
                    inpainted_file=escape_jsx_path(inpainted_path)
                )
                logger.info(f"Falling back to the working directory for the PSD inpainted image: {inpainted_path}")
            else:
                logger.debug(f"No usable inpainted image found: {inpainted_path}")
        
        # Mask layer - not added
        mask_layer_code = ""
        
        # Build the text layer code
        if default_font:
            logger.info(f"Using font for PSD export: {default_font}")
        else:
            logger.info("Using the Photoshop default font for PSD export")
        text_layers_code = ""
        if hasattr(ctx, 'text_regions') and ctx.text_regions:
            filtered_regions = [r for r in ctx.text_regions if r.translation]
            logger.info(f"Preparing to add {len(filtered_regions)} text layers to the PSD")
            for i, region in enumerate(filtered_regions):
                text_layers_code += generate_text_layer_jsx(i, region, default_font, line_spacing)
        
        # Build the complete JSX script
        # Path escaping: always forward slashes, with single quotes escaped, so the JSX string is not broken.
        jsx_script = JSX_TEMPLATE.format(
            input_file=escape_jsx_path(input_file),
            output_file=escape_jsx_path(output_file),
            error_file=escape_jsx_path(error_file),
            inpainted_layer_code=inpainted_layer_code,
            mask_layer_code=mask_layer_code,
            text_layers_code=text_layers_code,
        )
        
        # Save the JSX script (UTF-8 with BOM, so Photoshop reads non-ASCII text correctly)
        with open(jsx_file, 'w', encoding='utf-8-sig') as f:
            f.write(jsx_script)
        
        logger.info(f"Generated JSX script: {jsx_file}")
        
        # When verbose mode or script_only mode is on
        saved_script_path = None
        if verbose or script_only:
            try:
                image_name = os.path.basename(image_path) if image_path else "unknown"
                base_name, _ = os.path.splitext(image_name)

                if script_only and image_path:
                    # In script_only mode, save to manga_translator_work/psd
                    image_dir = os.path.dirname(os.path.abspath(image_path))
                    psd_dir = os.path.join(image_dir, 'manga_translator_work', 'psd')
                    os.makedirs(psd_dir, exist_ok=True)
                    debug_jsx_path = os.path.join(psd_dir, f"{base_name}_photoshop_script.jsx")
                elif result_path_fn:
                    # In verbose mode, or without image_path, save to the result folder
                    debug_jsx_path = result_path_fn(f"{base_name}_photoshop_script.jsx")
                else:
                    debug_jsx_path = None

                if debug_jsx_path:
                    with open(debug_jsx_path, 'w', encoding='utf-8') as f:
                        f.write(jsx_script)
                    saved_script_path = debug_jsx_path
                    logger.info(f"📝 JSX script saved: {debug_jsx_path}")
            except Exception as e:
                logger.warning(f"Failed to save JSX script: {e}")
        
        # When only the script is wanted, return here
        if script_only:
            logger.info("✅ Script generation only: JSX script saved, skipping Photoshop execution")
            if saved_script_path:
                logger.info(f"   Script path: {saved_script_path}")
            return
        
        # Run Photoshop
        ps_executable = find_photoshop_executable()
        if not ps_executable:
            raise FileNotFoundError(
                'Photoshop executable not found. Ensure Photoshop is installed or set PHOTOSHOP_PATH to the Photoshop.exe path.'
            )
        
        logger.info(f"Using Photoshop: {ps_executable}")
        logger.info(f"Executing script: {jsx_file}")
        
        # Run Photoshop (without waiting for the process to exit; only for the PSD file to appear)
        import time
        
        # Record the modification time of the output file (when it already exists)
        old_mtime = os.path.getmtime(output_file) if os.path.exists(output_file) else 0
        
        # Start Photoshop (without waiting)
        process = subprocess.Popen(
            [ps_executable, '-r', jsx_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Poll for the PSD file (wait at most 300 seconds)
        timeout = 300
        poll_interval = 0.5
        elapsed = 0
        
        while elapsed < timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval
            
            # Check whether the PSD file was created or updated
            if os.path.exists(output_file):
                new_mtime = os.path.getmtime(output_file)
                if new_mtime > old_mtime and os.path.getsize(output_file) > 0:
                    # The file exists; wait a moment to make sure writing has finished
                    time.sleep(0.5)
                    break
        
        # Read the output (non-blocking)
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stdout:
                stdout_text = stdout.decode('utf-8', errors='replace')
                logger.info(f"Photoshop output:\n{stdout_text}")
            if stderr:
                stderr_text = stderr.decode('utf-8', errors='replace')
                logger.warning(f"Photoshop error output:\n{stderr_text}")
        except subprocess.TimeoutExpired:
            # Photoshop is still running, which is normal
            pass
            
        # Check for a script error report
        if os.path.exists(error_file):
            with open(error_file, 'r') as f:
                error_msg = f.read()
            logger.error(f"Photoshop script execution error: {error_msg}")
            raise RuntimeError(f'Photoshop script error: {error_msg}')
        
        # Check whether the PSD file was created successfully
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info(f"PSD file generated: {output_file}")
        else:
            if elapsed >= timeout:
                raise RuntimeError(f'Photoshop execution timed out ({timeout}s); PSD file was not generated')
            else:
                raise RuntimeError(f'Failed to generate PSD file: {output_file}')
        
    finally:
        # Remove the temporary files
        for temp_file in [inpainted_file, mask_file, error_file]:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file}: {e}")
        
        # Outside verbose mode and script_only mode, delete the JSX script
        if not verbose and not script_only and os.path.exists(jsx_file):
            try:
                os.unlink(jsx_file)
            except Exception as e:
                logger.warning(f"Failed to delete JSX script {jsx_file}: {e}")


def _normalize_photoshop_path(value) -> Optional[str]:
    """Accept a quoted executable path or an install folder, and confirm that the target is a file."""
    if not isinstance(value, str):
        return None
    path = value.strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in ('"', "'"):
        path = path[1:-1]
    if not path:
        return None
    path = os.path.expanduser(os.path.expandvars(path))
    if platform.system() == "Windows" and os.path.isdir(path):
        path = os.path.join(path, "Photoshop.exe")
    return path if os.path.isfile(path) else None


def _find_photoshop_from_environment() -> Optional[str]:
    """Check the process environment and the variables saved in Windows, so a system variable changed after start-up is still seen."""
    value = os.getenv("PHOTOSHOP_PATH")
    if value:
        path = _normalize_photoshop_path(value)
        if path:
            logger.info(f"Found Photoshop via environment variable: {path}")
            return path
        logger.warning(f"The file specified by PHOTOSHOP_PATH does not exist: {value!r}")

    if platform.system() != "Windows":
        return None
    try:
        import winreg
    except ImportError:
        return None

    # os.environ is a snapshot from process start; a value just saved in the system settings may not be inherited yet.
    for hkey, subkey, source in [
        (winreg.HKEY_CURRENT_USER, r"Environment", 'user environment variable'),
        (winreg.HKEY_LOCAL_MACHINE,
         r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", 'system environment variable'),
    ]:
        try:
            with winreg.OpenKey(hkey, subkey) as key:
                value, _ = winreg.QueryValueEx(key, "PHOTOSHOP_PATH")
            path = _normalize_photoshop_path(value)
            if path:
                logger.info(f"Found Photoshop via {source}: {path}")
                return path
            logger.warning(f"The file specified by {source} PHOTOSHOP_PATH does not exist: {value!r}")
        except OSError as e:
            logger.debug(f"Failed to read {source} PHOTOSHOP_PATH: {e}")
    return None


def find_photoshop_from_registry() -> Optional[str]:
    """
    Find the Photoshop install path in the Windows registry

    Returns:
        The path of the Photoshop executable, or None when it is not found
    """
    if platform.system() != "Windows":
        return None
    
    try:
        import winreg
    except ImportError:
        logger.warning("Cannot import winreg, skipping registry lookup")
        return None
    
    # The default value of App Paths points directly at the executable and does not depend on the install drive.
    for hkey in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
        for view in [winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY]:
            try:
                with winreg.OpenKey(
                    hkey, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Photoshop.exe",
                    0, winreg.KEY_READ | view,
                ) as key:
                    value, _ = winreg.QueryValueEx(key, "")
                ps_exe = _normalize_photoshop_path(value)
                if ps_exe:
                    logger.info(f"Found Photoshop via the App Paths registry: {ps_exe}")
                    return ps_exe
            except OSError:
                continue

    # Possible registry paths
    registry_paths = [
        # Photoshop CC and newer versions
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Adobe\Photoshop"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Adobe\Photoshop"),
        # Path of a 32-bit program on a 64-bit system
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Adobe\Photoshop"),
    ]
    
    for hkey, subkey_path in registry_paths:
        try:
            # Open the Photoshop main key
            with winreg.OpenKey(hkey, subkey_path) as key:
                # Enumerate all version subkeys
                i = 0
                versions = []
                while True:
                    try:
                        version_key = winreg.EnumKey(key, i)
                        versions.append(version_key)
                        i += 1
                    except OSError:
                        break
                
                # Sort by version number, descending (the newest version first)
                versions.sort(reverse=True)
                
                # Try each version
                for version in versions:
                    try:
                        version_path = f"{subkey_path}\\{version}"
                        with winreg.OpenKey(hkey, version_path) as version_key:
                            # Try to read ApplicationPath or InstallPath
                            for value_name in ["ApplicationPath", "InstallPath", "Path"]:
                                try:
                                    install_path, _ = winreg.QueryValueEx(version_key, value_name)
                                    ps_exe = _normalize_photoshop_path(install_path)
                                    if ps_exe:
                                        logger.info(f"Found Photoshop via the registry: {ps_exe}")
                                        return ps_exe
                                except FileNotFoundError:
                                    continue
                    except Exception as e:
                        logger.debug(f"Failed to read registry version {version}: {e}")
                        continue
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.debug(f"Failed to read registry path {subkey_path}: {e}")
            continue
    
    return None


def find_photoshop_executable() -> Optional[str]:
    """
    Find the path of the Photoshop executable

    Search order:
    1. The PHOTOSHOP_PATH environment variable (including the user and system variables currently saved in Windows)
    2. The Windows registry (Windows only)
    3. Common install paths
    4. Walking the Adobe folder

    Returns:
        The full path of the Photoshop executable, or None when it is not found
    """
    
    # 1. Prefer the environment variable
    ps_path = _find_photoshop_from_environment()
    if ps_path:
        return ps_path
    
    system = platform.system()
    
    if system == "Windows":
        # 2. Look in the registry (the most reliable)
        ps_path = find_photoshop_from_registry()
        if ps_path:
            return ps_path
        
        # 3. Common install paths on Windows
        possible_paths = [
            r"C:\Program Files\Adobe\Adobe Photoshop 2024\Photoshop.exe",
            r"C:\Program Files\Adobe\Adobe Photoshop 2023\Photoshop.exe",
            r"C:\Program Files\Adobe\Adobe Photoshop 2022\Photoshop.exe",
            r"C:\Program Files\Adobe\Adobe Photoshop 2021\Photoshop.exe",
            r"C:\Program Files\Adobe\Adobe Photoshop CC 2019\Photoshop.exe",
            r"C:\Program Files\Adobe\Adobe Photoshop CC 2018\Photoshop.exe",
        ]
        
        # Check Program Files (x86) as well
        program_files_x86 = os.getenv("ProgramFiles(x86)")
        if program_files_x86:
            for path in list(possible_paths):
                x86_path = path.replace(r"C:\Program Files", program_files_x86)
                possible_paths.append(x86_path)
        
        # Search every possible path
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found Photoshop in a common location: {path}")
                return path
        
        # 4. Walk the Adobe folder in Program Files
        for program_files_var in ["ProgramFiles", "ProgramFiles(x86)"]:
            program_files = os.getenv(program_files_var)
            if not program_files:
                continue
            
            adobe_dir = os.path.join(program_files, "Adobe")
            if os.path.exists(adobe_dir):
                try:
                    folders = sorted(os.listdir(adobe_dir), reverse=True)  # Descending, newer versions first
                    for folder in folders:
                        if "Photoshop" in folder:
                            ps_exe = os.path.join(adobe_dir, folder, "Photoshop.exe")
                            if os.path.exists(ps_exe):
                                logger.info(f"Found Photoshop in the Adobe directory: {ps_exe}")
                                return ps_exe
                except Exception as e:
                    logger.debug(f"Failed to scan the Adobe directory: {e}")
    
    elif system == "Darwin":  # macOS
        possible_paths = [
            "/Applications/Adobe Photoshop 2024/Adobe Photoshop 2024.app/Contents/MacOS/Adobe Photoshop 2024",
            "/Applications/Adobe Photoshop 2023/Adobe Photoshop 2023.app/Contents/MacOS/Adobe Photoshop 2023",
            "/Applications/Adobe Photoshop 2022/Adobe Photoshop 2022.app/Contents/MacOS/Adobe Photoshop 2022",
            "/Applications/Adobe Photoshop 2021/Adobe Photoshop 2021.app/Contents/MacOS/Adobe Photoshop 2021",
            "/Applications/Adobe Photoshop CC 2019/Adobe Photoshop CC 2019.app/Contents/MacOS/Adobe Photoshop CC 2019",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found Photoshop in a common location: {path}")
                return path
    
    logger.warning("Photoshop installation not found")
    return None


def test_photoshop_installation() -> bool:
    """
    Test whether Photoshop is installed correctly and usable

    Returns:
        True when Photoshop is usable, otherwise False
    """
    ps_exe = find_photoshop_executable()
    if not ps_exe:
        logger.error("Photoshop installation not found")
        return False
    
    logger.info(f"Found Photoshop: {ps_exe}")
    return True
