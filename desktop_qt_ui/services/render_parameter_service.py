"""
Render parameter management service.
Provides calculation, customisation, storage and management of font and layout parameters
"""
import copy
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

VALID_LAYOUT_MODES = {"smart_scaling", "strict", "balloon_fill"}


class Alignment(Enum):
    """Alignment enumeration"""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    AUTO = "auto"

class Direction(Enum):
    """Text direction enumeration"""
    HORIZONTAL = "h"
    VERTICAL = "v"
    AUTO = "auto"


def _normalize_direction(value: Any) -> str:
    """Unify the layout direction; old reversed values are only aliases, and the reading order is decided by the language."""
    direction = str(getattr(value, "value", value) or "auto").strip().lower()
    return {
        "horizontal": "h",
        "vertical": "v",
        "hr": "h",
        "vr": "v",
    }.get(direction, direction if direction in {"h", "v", "auto"} else "auto")


@dataclass
class RenderParameters:
    """Data class of the render parameters"""
    # Font parameters
    font_size: int = 12
    font_family: str = ""

    # Colour parameters
    fg_color: Tuple[int, int, int] = (255, 255, 255)  # Foreground colour
    bg_color: Tuple[int, int, int] = (0, 0, 0)  # Background colour / stroke colour
    opacity: float = 1.0  # Opacity
    
    # Layout parameters
    alignment: str = "center"
    direction: str = "auto"
    line_spacing: float = 1.0  # Line spacing multiplier
    letter_spacing: float = 1.0  # Letter spacing multiplier
    layout_mode: str = "smart_scaling"  # Layout mode
    balloon_fill_mask_layout: bool = False  # Break lines by the bubble mask, enlarge the font and avoid collisions across bubbles
    disable_auto_wrap: bool = False  # Disable automatic wrapping (AI line breaking)
    font_size_offset: int = 0  # Font size offset
    font_size_minimum: int = 0  # Minimum font size
    max_font_size: int = 0  # Maximum font size
    font_scale_ratio: float = 1.0  # Font scale ratio
    center_text_in_bubble: bool = False  # Centre the text when AI line breaking is on
    optimize_line_breaks: bool = False  # Optimise line breaks automatically
    semantic_linebreak: bool = False  # Semantic line breaking for Chinese
    remove_linebreak_punctuation: bool = False  # Remove commas and periods around line breaks
    strict_smart_scaling: bool = False  # Do not enlarge the text box when AI line breaking enlarges the text automatically

    # Effect parameters
    stroke_width: float = 0.07  # Stroke width (as a ratio of the font size)
    shadow_radius: float = 0.0  # Shadow radius
    shadow_strength: float = 1.0  # Shadow strength
    shadow_color: Tuple[int, int, int] = (0, 0, 0)  # Shadow colour
    shadow_offset: List[float] = None  # Shadow offset
    
    # Render options
    hyphenate: bool = True  # Whether hyphenation is on
    disable_font_border: bool = False  # Whether the font border is disabled
    
    def __post_init__(self):
        self.direction = _normalize_direction(self.direction)
        if self.shadow_offset is None:
            self.shadow_offset = [0.0, 0.0]
        if self.layout_mode not in VALID_LAYOUT_MODES:
            raise ValueError(
                f"Invalid layout_mode: {self.layout_mode!r}. "
                f"Supported values: {', '.join(sorted(VALID_LAYOUT_MODES))}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary form"""
        return asdict(self)

    @property
    def effective_stroke_width(self) -> float:
        """The one effective width used by both the measuring and the drawing of the stroke."""
        if self.disable_font_border:
            return 0.0
        return max(float(self.stroke_width), 0.0)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RenderParameters':
        """Create a parameter object from a dictionary"""
        clean = dict(data or {})
        return cls(**clean)

@dataclass
class ParameterPreset:
    """Parameter preset"""
    name: str
    description: str
    parameters: RenderParameters

class RenderParameterService:
    """Render parameter management service"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        from services import get_config_service
        self.config_service = get_config_service()

        # Custom parameters of each region
        self.region_parameters: Dict[int, RenderParameters] = {}

        # No separate default parameters are kept any more; the configuration service is used directly
        # Preset parameters
        self.presets: Dict[str, ParameterPreset] = {}
        self._init_default_presets()


    def get_default_parameters(self) -> RenderParameters:
        """Get the default parameters from the current configuration service"""
        config = self.config_service.get_config()
        render_fields = RenderParameters.__dataclass_fields__.keys()
        global_render_config = config.render.model_dump()
        # Leave out None values, so the dataclass defaults apply
        valid_global_config = {k: v for k, v in global_render_config.items() if k in render_fields}
        return RenderParameters(**valid_global_config)

    def _init_default_presets(self):
        # Standard manga preset
        self.presets["manga_standard"] = ParameterPreset(
            name="漫画标准",
            description="适合大部分漫画的标准设置",
            parameters=RenderParameters(
                font_size=16,
                alignment="center",
                direction="auto",
                line_spacing=1.2,
                letter_spacing=1.0,
                fg_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                stroke_width=0.15
            )
        )
        
        # Light novel preset
        self.presets["novel_standard"] = ParameterPreset(
            name="轻小说标准",
            description="适合轻小说的横排文本设置",
            parameters=RenderParameters(
                font_size=14,
                alignment="left",
                direction="h",
                line_spacing=1.4,
                letter_spacing=1.1,
                fg_color=(0, 0, 0),
                bg_color=(255, 255, 255),
                stroke_width=0.0
            )
        )
        
        # Classical literature preset
        self.presets["classical_vertical"] = ParameterPreset(
            name="古典竖排",
            description="适合古典文学的竖排文本设置",
            parameters=RenderParameters(
                font_size=18,
                alignment="right",
                direction="v",
                line_spacing=1.0,
                letter_spacing=0.9,
                fg_color=(0, 0, 0),
                bg_color=(255, 255, 255),
                stroke_width=0.0
            )
        )
    
    def calculate_default_parameters(self, region_data: Dict[str, Any]) -> RenderParameters:
        """Compute the default render parameters from the original text box"""
        params = self.get_default_parameters()
        try:
            lines = region_data.get('lines', [])
            if not lines or not lines[0]:
                return self._apply_region_overrides(params, region_data)

            all_points = [point for poly in lines for point in poly]
            if len(all_points) < 4:
                return self._apply_region_overrides(params, region_data)
            
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            # Font size (80% of the region height)
            if height > 0:
                font_size = max(int(height * 0.6), 8)  # At least 8 pixels
                font_size = min(font_size, 72)  # At most 72 pixels
            else:
                font_size = 12
            
            # Decide the text direction
            aspect_ratio = width / height if height > 0 else 1.0
            if aspect_ratio > 2.0:
                direction = "h"  # Clearly horizontal
                alignment = "center"
            elif aspect_ratio < 0.5:
                direction = "v"  # Clearly vertical
                alignment = "right"
            else:
                direction = "auto"  # Decide automatically
                alignment = "center"
            
            params.font_size = font_size
            params.alignment = alignment
            params.direction = direction

            self.logger.debug(f"Calculating default parameters: size={width}x{height}, font={font_size}, direction={direction}")
            return self._apply_region_overrides(params, region_data)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate default parameters: {e}")
            return self._apply_region_overrides(params, region_data)

    @staticmethod
    def _parse_hex_color(value: Any):
        if not (isinstance(value, str) and value.startswith('#') and len(value) == 7):
            return None
        try:
            return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))
        except ValueError:
            return None

    def _apply_region_overrides(
        self,
        params: RenderParameters,
        region_data: Dict[str, Any] = None,
    ) -> RenderParameters:
        """Apply canonical region fields once; callers always receive a copy.

        ``stroke_width`` is the only accepted stroke-width field.  Measurement
        and drawing both consume the returned ``RenderParameters`` instance,
        so aliases cannot silently diverge again.
        """
        resolved = copy.deepcopy(params)
        resolved.direction = _normalize_direction(resolved.direction)
        if not region_data:
            return resolved

        for field_name in RenderParameters.__dataclass_fields__:
            if field_name not in region_data:
                continue
            value = region_data[field_name]
            if value is None:
                continue
            if field_name == 'font_size' and not value:
                continue
            if field_name in {'font_family', 'alignment', 'direction'} and not value:
                continue
            if field_name == 'stroke_width':
                value = max(float(value), 0.0)
            elif field_name == 'direction':
                value = _normalize_direction(value)
            elif field_name in {'fg_color', 'bg_color'} and isinstance(value, list):
                value = tuple(value)
            setattr(resolved, field_name, value)

        font_color = self._parse_hex_color(region_data.get('font_color'))
        if font_color is not None:
            resolved.fg_color = font_color
        elif region_data.get('fg_colors') is not None:
            resolved.fg_color = tuple(region_data['fg_colors'])

        if region_data.get('bg_colors') is not None:
            resolved.bg_color = tuple(region_data['bg_colors'])

        return resolved
    
    def get_region_parameters(self, region_index: int, region_data: Dict[str, Any] = None) -> RenderParameters:
        """Resolve one immutable-by-convention parameter snapshot for a region."""
        if region_index in self.region_parameters:
            base = self.region_parameters[region_index]
        elif region_data:
            base = self.calculate_default_parameters(region_data)
            self.region_parameters[region_index] = copy.deepcopy(base)
        else:
            base = self.get_default_parameters()
        return self._apply_region_overrides(base, region_data)
    
    def set_region_parameters(self, region_index: int, parameters: RenderParameters):
        """Set the render parameters of the given region"""
        self.region_parameters[region_index] = copy.deepcopy(parameters)
        self.region_parameters[region_index].direction = _normalize_direction(parameters.direction)
        self.logger.debug(f"Setting rendering parameters for region {region_index}")
    
    def update_region_parameter(self, region_index: int, param_name: str, value: Any):
        """Update a single parameter of the given region"""
        if region_index not in self.region_parameters:
            self.region_parameters[region_index] = self.get_default_parameters()

        if hasattr(self.region_parameters[region_index], param_name):
            if param_name == 'direction':
                value = _normalize_direction(value)
            setattr(self.region_parameters[region_index], param_name, value)
            self.logger.debug(f"Updating region {region_index} parameter {param_name} = {value}")
        else:
            self.logger.warning(f"Unknown parameter: {param_name}")
    
    def apply_preset(self, region_index: int, preset_name: str) -> bool:
        """Apply a parameter preset to the given region"""
        if preset_name not in self.presets:
            self.logger.warning(f"Preset not found: {preset_name}")
            return False
        
        preset_params = copy.deepcopy(self.presets[preset_name].parameters)
        self.set_region_parameters(region_index, preset_params)
        self.logger.info(f"Applying preset '{preset_name}' to region {region_index}")
        return True
    
    def create_custom_preset(self, name: str, description: str, parameters: RenderParameters):
        """Create a custom preset"""
        self.presets[name] = ParameterPreset(
            name=name,
            description=description,
            parameters=copy.deepcopy(parameters)
        )
        self.logger.info(f"Creating custom preset: {name}")
    
    def get_preset_list(self) -> List[Dict[str, str]]:
        """Get the list of presets"""
        return [
            {
                "name": preset.name,
                "key": key,
                "description": preset.description
            }
            for key, preset in self.presets.items()
        ]
    
    def export_parameters_for_backend(self, region_index: int, region_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export the parameters for the backend to recognise and apply"""
        params = self.get_region_parameters(region_index, region_data)
        
        # Convert to the format the backend understands
        font_to_use = params.font_family
        if not font_to_use:
            default_params = self.get_default_parameters()
            font_to_use = default_params.font_family
        
        backend_params = {
            # Font parameters
            'font_size': params.font_size,
            'font_family': font_to_use,

            # Colour parameters
            'font_color': f"#{params.fg_color[0]:02x}{params.fg_color[1]:02x}{params.fg_color[2]:02x}" if isinstance(params.fg_color, (list, tuple)) and len(params.fg_color) == 3 else params.fg_color,
            'opacity': params.opacity,
            
            # Layout parameters
            'alignment': params.alignment,
            'direction': {'h': 'horizontal', 'v': 'vertical'}.get(params.direction, 'auto'),
            'vertical': params.direction == 'v',
            'line_spacing': params.line_spacing,
            'letter_spacing': params.letter_spacing,
            
            # Effect parameters
            'stroke_width': params.effective_stroke_width,
            'shadow_radius': params.shadow_radius,
            'shadow_strength': params.shadow_strength,
            'shadow_color': params.shadow_color,
            'shadow_offset': params.shadow_offset,
            
            # Render options
            'hyphenate': params.hyphenate,
            'disable_font_border': params.disable_font_border,
            'disable_auto_wrap': params.disable_auto_wrap,
            'layout_mode': params.layout_mode,
            'balloon_fill_mask_layout': params.balloon_fill_mask_layout,
            'font_size_offset': params.font_size_offset,
            'font_size_minimum': params.font_size_minimum,
            'max_font_size': params.max_font_size,
            'font_scale_ratio': params.font_scale_ratio,
            'center_text_in_bubble': params.center_text_in_bubble,
            'semantic_linebreak': params.semantic_linebreak,
            'remove_linebreak_punctuation': params.remove_linebreak_punctuation,
            # Add the metadata
            '_render_params_version': '1.0',
            '_generated_by': 'desktop-ui'
        }
        
        # Stroke colour - params.bg_color is used as the stroke colour
        backend_params['text_stroke_color'] = params.bg_color

        return backend_params
    
    def import_parameters_from_json(self, region_index: int, json_data: Dict[str, Any]):
        """Import parameters from JSON data"""
        try:
            # Keep the valid parameters only
            valid_params = {}
            param_fields = RenderParameters.__dataclass_fields__.keys()
            
            for key, value in json_data.items():
                # Special handling for the inconsistent colour key names (plural in the JSON, singular in the dataclass)
                if key == 'fg_colors':
                    key = 'fg_color'
                    value = tuple(value) if isinstance(value, list) else value
                elif key == 'bg_colors':
                    key = 'bg_color'
                    value = tuple(value) if isinstance(value, list) else value
                elif key == 'text_stroke_color':
                    key = 'bg_color'
                    # Handle a colour in hex format
                    if isinstance(value, str) and value.startswith('#'):
                        try:
                            r = int(value[1:3], 16)
                            g = int(value[3:5], 16)
                            b = int(value[5:7], 16)
                            value = (r, g, b)
                        except ValueError:
                            continue
                    else:
                         value = tuple(value) if isinstance(value, list) else value
                elif key == 'font_color':
                    # Handle a colour in hex format
                    if isinstance(value, str) and value.startswith('#'):
                        try:
                            r = int(value[1:3], 16)
                            g = int(value[3:5], 16)
                            b = int(value[5:7], 16)
                            key = 'fg_color'
                            value = (r, g, b)
                        except ValueError:
                            continue

                if key in param_fields:
                    valid_params[key] = value
            
            if valid_params:
                # Take the default parameters of the configuration service as the base first
                base_params = self.get_default_parameters()
                base_dict = base_params.to_dict()

                # Override with the values from the JSON (line_spacing is skipped)
                base_dict.update(valid_params)

                params = RenderParameters(**base_dict)
                self.set_region_parameters(region_index, params)
                self.logger.debug(f"Importing parameters for region {region_index} from JSON")
                return True
            else:
                self.logger.warning("No valid rendering parameters in JSON")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to import parameters: {e}")
            return False
    
    def batch_update_parameters(self, updates: Dict[int, Dict[str, Any]]):
        """Update the parameters of several regions as a batch"""
        for region_index, param_updates in updates.items():
            for param_name, value in param_updates.items():
                self.update_region_parameter(region_index, param_name, value)
    
    def copy_parameters(self, from_region: int, to_region: int):
        """Copy the parameters from one region to another"""
        if from_region in self.region_parameters:
            source_params = copy.deepcopy(self.region_parameters[from_region])
            self.set_region_parameters(to_region, source_params)
            self.logger.info(f"Copying parameters from region {from_region} to region {to_region}")
            return True
        return False
    
    def reset_region_parameters(self, region_index: int):
        """Reset the parameters of a region to the defaults"""
        if region_index in self.region_parameters:
            del self.region_parameters[region_index]
            self.logger.info(f"Resetting parameters for region {region_index}")

    def clear_cache(self):
        """Clear the cache of custom parameters of all regions"""
        self.region_parameters.clear()
    
    def get_parameter_summary(self, region_index: int) -> Dict[str, str]:
        """Get summary information of the parameters"""
        params = self.get_region_parameters(region_index)
        
        direction_map = {
            "h": "水平",
            "v": "垂直", 
            "auto": "自动"
        }
        
        alignment_map = {
            "left": "左对齐",
            "center": "居中",
            "right": "右对齐",
            "auto": "自动"
        }
        
        return {
            "字体大小": f"{params.font_size}px",
            "对齐方式": alignment_map.get(params.alignment, params.alignment),
            "文本方向": direction_map.get(params.direction, params.direction),
            "行间距": f"{params.line_spacing:.1f}倍",
            "字间距": f"{params.letter_spacing:.1f}倍",
            "描边宽度": f"{params.stroke_width:.2f}",
            "前景色": f"RGB{params.fg_color}",
            "背景色": f"RGB{params.bg_color}"
        }

# Instantiation is handled by the ServiceContainer in services/__init__.py
