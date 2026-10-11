import copy
import os
import weakref
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from manga_translator.translators.manga_context import load_chapter_context
from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot

from editor.commands import _NO_MASK_CHANGE, MoveRegionCommand, UpdateRegionCommand
from editor.geometry_commit_pipeline import build_rotate_region_data
from editor.region_geometry_state import RegionGeometryState
from services import (
    get_async_service,
    get_config_service,
    get_file_service,
    get_history_service,
    get_i18n_manager,
    get_logger,
    get_ocr_service,
    get_render_parameter_service,
    get_resource_manager,
    get_translation_service,
)

from .controller_document_service import EditorControllerDocumentService
from .controller_export_service import (
    EditorControllerExportService,
    ExportOutcome,
)
from .controller_inpaint_service import EditorControllerInpaintService
from .editor_model import EditorModel
from .image_utils import image_like_to_display_array
from .render_text_value import has_renderable_text, render_text_value_from_region

_UNSET = object()

# Weak-reference registry of live controllers: the exit path (app_logic.shutdown) has to find the controllers for thread pool
# clean-up without holding a reference to the editor; weak references avoid extending their lifetime.
_ACTIVE_CONTROLLERS: "weakref.WeakSet" = weakref.WeakSet()


def get_active_editor_controllers() -> list:
    """Return the list of EditorController instances that are still alive (for clean-up on exit)."""
    return list(_ACTIVE_CONTROLLERS)


@dataclass(slots=True)
class _AsyncRegionUpdateRequest:
    """Request for writing a background OCR or translation result to the model on the main thread.

    updates holds stable region_id values, not indexes. An index can only be resolved on the main thread, right before the write.
    """

    field_name: str
    updates: list[tuple[Optional[int], str]]
    task_kind: str
    error_count: int = 0


# Changing these fields affects the pixel size of the text that the font size is derived from, so the white box has to be refreshed too:
# anchor the body centre and update width and height to the full paint size given by calc_box_from_font(new parameters).
_FONT_AFFECTING_FIELDS = frozenset(
    {
        "translation",
        "translation_rich",
        "text",
        "font_size",
        "font_family",
        "letter_spacing",
        "line_spacing",
        "direction",
        "stroke_width",
        "disable_font_border",
    }
)

_STYLE_PATCH_FIELDS = frozenset(
    {
        "font_size",
        "font_family",
        "font_color",
        "stroke_color",
        "stroke_width",
        "line_spacing",
        "letter_spacing",
        "angle",
        "alignment",
        "direction",
    }
)


def _sync_white_frame_size_for_font_change(
    region_data: dict,
    old_region_data: Optional[dict],
    render_params,
    old_render_params,
) -> None:
    """After properties such as font, translation, stroke or letter spacing change, sync the white box size to the size derived from the font size.

    The white box is still the render box (with decorations outside the body, such as ruby); what is anchored is the body centre:
    new box centre = (old box centre + old body offset) − new body offset,
    where offset = the body centre returned by calc_box_from_font − the exact centre of the render box (coordinates inside the box).
    For plain text both offsets are zero and the behaviour is exactly "keep the box centre"; when ruby is added to or removed from rich text,
    the body stays pinned and the render box grows or shrinks on the side of the decoration.
    has_custom_white_frame=True is set so that it takes precedence over render_box in deciding the render centre.
    """
    try:
        from manga_translator.rendering import calc_box_from_font

        def _box_metrics(data: dict, params):
            """Returns (box width, box height, body offset); None when the text or the font size is invalid."""
            font_size = int(
                data.get("font_size") or getattr(params, "font_size", 0) or 0
            )
            value = render_text_value_from_region(data)
            if font_size <= 0 or not has_renderable_text(value):
                return None
            direction = data.get("direction") or getattr(params, "direction", "h")
            is_horizontal = direction in ("h", "horizontal", "hr")
            line_spacing = float(getattr(params, "line_spacing", 1.0) or 1.0)
            letter_spacing = float(getattr(params, "letter_spacing", 1.0) or 1.0)
            w, h, _, (body_x, body_y) = calc_box_from_font(
                font_size,
                value,
                is_horizontal,
                line_spacing,
                None,
                None,
                center=None,
                angle=0,
                letter_spacing=letter_spacing,
                stroke_width=params.effective_stroke_width,
            )
            if w <= 0 or h <= 0:
                return None
            return (
                float(w),
                float(h),
                (float(body_x) - w / 2.0, float(body_y) - h / 2.0),
            )

        new_metrics = _box_metrics(region_data, render_params)
        if new_metrics is None:
            return
        w, h, new_delta = new_metrics

        wf = region_data.get("white_frame_rect_local")
        if isinstance(wf, (list, tuple)) and len(wf) == 4:
            local_cx = (float(wf[0]) + float(wf[2])) / 2.0
            local_cy = (float(wf[1]) + float(wf[3])) / 2.0
        else:
            local_cx = local_cy = 0.0

        # Body anchor = centre of the old box + the old body offset.
        # When the old text is not available, treat the offset as unchanged (which falls back to the old behaviour of keeping the box centre).
        old_metrics = (
            _box_metrics(old_region_data, old_render_params)
            if old_region_data
            else None
        )
        old_delta = old_metrics[2] if old_metrics is not None else new_delta
        new_cx = (local_cx + old_delta[0]) - new_delta[0]
        new_cy = (local_cy + old_delta[1]) - new_delta[1]

        half_w, half_h = w / 2.0, h / 2.0
        region_data["white_frame_rect_local"] = [
            new_cx - half_w,
            new_cy - half_h,
            new_cx + half_w,
            new_cy + half_h,
        ]
        region_data["has_custom_white_frame"] = True
    except Exception:
        return


class EditorController(QObject):
    """
    Editor controller (Controller)

    Handles all business logic and user interaction of the editor.
    It responds to signals from the view (View), calls services (Service) to do the work and updates the model (Model).
    """

    # Signal for thread-safe model updates
    _update_display_mask_type = pyqtSignal(str)
    _regions_update_finished = pyqtSignal(object)
    _ocr_finished = pyqtSignal(str, str)
    _translation_finished = pyqtSignal(str, str)
    _inpaint_result_ready = pyqtSignal(object)

    # Export queue worker -> GUI thread signals
    _export_queue_status_signal = pyqtSignal(object)
    _export_job_finished_signal = pyqtSignal(object)

    _load_result_ready = pyqtSignal(object)  # Signal for load results
    _deferred_load_requested = pyqtSignal(str)

    def __init__(self, model: EditorModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.view = None  # Set later in EditorView
        self.logger = get_logger(__name__)

        # Get the services needed
        self.ocr_service = get_ocr_service()
        self.translation_service = get_translation_service()
        self.async_service = get_async_service()
        self.history_service = get_history_service()  # For undo/redo
        self.file_service = get_file_service()
        self.config_service = get_config_service()
        self.resource_manager = get_resource_manager()  # The new resource manager

        self.document_service = EditorControllerDocumentService(self)
        self.inpaint_service = EditorControllerInpaintService(self)
        self.export_service = EditorControllerExportService(self)
        self._export_status_text = ""
        self._export_toast = None

        # Connect internal signals for thread-safe updates
        self._update_display_mask_type.connect(self.model.set_display_mask_type)
        self._regions_update_finished.connect(self.on_regions_update_finished)
        self._ocr_finished.connect(self._on_ocr_finished)
        self._translation_finished.connect(self._on_translation_finished)
        self._inpaint_result_ready.connect(
            self._apply_inpaint_result,
            type=Qt.ConnectionType.QueuedConnection,
        )
        self._load_result_ready.connect(self._apply_load_result)  # Connect the load result signal
        self._deferred_load_requested.connect(self.document_service.do_load_image)
        self._export_queue_status_signal.connect(self._on_export_queue_status_changed)
        self._export_job_finished_signal.connect(self._on_export_job_finished)

        self.model.effective_mask_delta_changed.connect(
            self.inpaint_service.on_effective_mask_delta_changed
        )
        self.history_service.undo_redo_state_changed.connect(
            self._on_history_undo_redo_state_changed
        )

        _ACTIVE_CONTROLLERS.add(self)

    def shutdown(self) -> None:
        """Stop cancellable editor work, then drain the durable export queue."""
        assistant = getattr(self.view, 'chapter_assistant', None)
        if assistant is not None:
            assistant.shutdown()
        try:
            self.document_service.shutdown()
        except Exception as e:
            self.logger.warning(f"Editor document service shutdown failed: {e}")
        try:
            self.export_service.shutdown()
        except Exception as e:
            self.logger.warning(f"Editor export queue shutdown failed: {e}")

    # ========== Resource Access Helpers ==========

    @staticmethod
    def _normalize_image_path(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        return os.path.normcase(os.path.normpath(path))

    def _is_same_source_image(self, left: Optional[str], right: Optional[str]) -> bool:
        left_path = self._normalize_image_path(left)
        right_path = self._normalize_image_path(right)
        return bool(left_path and right_path and left_path == right_path)

    def _load_detached_image_array(
        self,
        image_path: str,
        target_size: tuple[int, int],
        *,
        resize: bool = True,
    ) -> np.ndarray:
        """Load an auxiliary image and normalise it to numpy directly, so it is not held as both PIL and ndarray."""
        detached_image = self.resource_manager.load_detached_image(image_path)
        resized_image = detached_image
        try:
            if resize and detached_image.size != target_size:
                resized_image = detached_image.resize(
                    target_size, Image.Resampling.LANCZOS
                )
            return image_like_to_display_array(resized_image, copy=False)
        finally:
            if resized_image is not detached_image:
                try:
                    resized_image.close()
                except Exception:
                    pass
            try:
                detached_image.close()
            except Exception:
                pass

    def _log_memory_snapshot(self, stage: str) -> None:
        try:
            self.resource_manager.log_memory_snapshot(stage, logger=self.logger)
        except Exception as e:
            self.logger.debug(f"Failed to log memory snapshot at {stage}: {e}")

    def _merge_live_geometry_state(self, region_index: int, region_data: dict) -> dict:
        """Keep the valid persistent geometry state of the current item for a style-only update."""
        if not isinstance(region_data, dict):
            return region_data

        try:
            gv = self.get_graphics_view()
            if gv is None:
                return region_data

            live_patch = gv.get_live_region_state_patch(region_index)
            if not live_patch:
                return region_data

            merged_region_data = copy.deepcopy(region_data)
            merged_region_data.update(live_patch)
            return merged_region_data
        except Exception:
            return region_data

    def _queue_async_region_updates(
        self,
        updates: list[tuple[Optional[int], str]],
        *,
        field_name: str,
        task_kind: str,
        error_count: int = 0,
    ) -> None:
        """Hand the result of an asynchronous task to the main thread, to be written back to the model by stable region_id.

        updates: [(region_id, value), ...]. The model is not read here and ids are not resolved to an
        index in advance; the main thread slot locates them again right before the write, so inserting or deleting
        a region while the result waits in the queue cannot make it hit the wrong target.
        """
        if not updates:
            return

        self._regions_update_finished.emit(
            _AsyncRegionUpdateRequest(
                field_name=field_name,
                updates=list(updates),
                task_kind=task_kind,
                error_count=error_count,
            )
        )

    def _finish_async_region_update(
        self,
        task_kind: str,
        *,
        applied_count: int,
        skipped_count: int,
        error_count: int = 0,
    ) -> None:
        if task_kind == "ocr":
            if applied_count > 0 and skipped_count == 0 and error_count == 0:
                self._ocr_finished.emit("success", "识别完成")
            elif applied_count > 0:
                self._ocr_finished.emit(
                    "warning",
                    f"识别部分完成，已应用 {applied_count} 项，跳过 {skipped_count + error_count} 项",
                )
            elif skipped_count > 0:
                self._ocr_finished.emit("warning", "识别结果未应用，目标区域已变化")
            elif error_count > 0:
                self._ocr_finished.emit("error", "识别失败")
            else:
                self._ocr_finished.emit("warning", "未识别到可更新的文本")
            return

        if task_kind == "translation":
            if applied_count > 0 and skipped_count == 0:
                self._translation_finished.emit("success", "翻译完成")
            elif applied_count > 0:
                self._translation_finished.emit(
                    "warning",
                    f"翻译部分完成，已应用 {applied_count} 项，跳过 {skipped_count} 项",
                )
            elif skipped_count > 0:
                self._translation_finished.emit(
                    "warning", "翻译结果未应用，目标区域已变化"
                )
            else:
                self._translation_finished.emit("warning", "未生成可应用的翻译结果")

    def _finalize_progress_toast(
        self, toast_attr: str, status: str, message: str
    ) -> None:
        toast = getattr(self, toast_attr, None)
        if toast is not None:
            try:
                toast.close()
            except Exception:
                pass
            setattr(self, toast_attr, None)

        toast_manager = getattr(self, "toast_manager", None)
        if toast_manager is None or not message:
            return

        if status == "success":
            toast_manager.show_success(message)
        elif status == "error":
            toast_manager.show_error(message)
        else:
            toast_manager.show_info(message)

    def get_graphics_view(self):
        return getattr(self.view, "graphics_view", None) if self.view else None

    def get_property_panel(self):
        return getattr(self.view, "property_panel", None) if self.view else None

    def get_toolbar(self):
        return getattr(self.view, "toolbar", None) if self.view else None

    def get_toast_manager(self):
        return getattr(self, "toast_manager", None)

    def commit_pending_edits(self) -> None:
        """Before the model is read for a persistence decision (dirty detection, export), commit the local drafts the view layer is holding.

        The only source at present is the debounce draft of the floating rich-text editor; any future control that "batches locally and
        writes the model later" should hook in here, instead of relying on each reading path to remember to flush.
        """
        editor = getattr(self.view, "rich_text_editor", None) if self.view else None
        if editor is None:
            return
        try:
            editor.flush_pending_changes()
        except Exception as e:
            self.logger.warning(f"commit_pending_edits failed: {e}")

    def set_compare_mode(self, enabled: bool) -> None:
        if self.view is None:
            return
        set_compare_mode = getattr(self.view, "set_compare_mode", None)
        if callable(set_compare_mode):
            set_compare_mode(enabled)

    def set_view(self, view):
        """Set the view reference, for updating the UI state"""
        self.view = view
        graphics_view = self.get_graphics_view()
        if graphics_view is not None:
            graphics_view.set_controller(self)
        # The toast manager and its signal connections are created once: reused when set_view is called again,
        # so a repeated connect does not show the same toast several times
        existing_toast_manager = getattr(self, "toast_manager", None)
        if (
            existing_toast_manager is None
            or getattr(existing_toast_manager, "parent", None) is not view
        ):
            from ui.widgets.toast_notification import ToastManager

            self.toast_manager = ToastManager(view)
        # Initialise the state of the undo/redo buttons
        self._update_undo_redo_buttons()

    def _close_export_progress_toast(self) -> None:
        toast = getattr(self, "_export_toast", None)
        if toast is not None:
            try:
                toast.close()
            except Exception:
                pass
        self._export_toast = None
        self._export_status_text = ""

    @pyqtSlot(object)
    def _on_export_queue_status_changed(self, unfinished_count: object) -> None:
        try:
            unfinished_count = int(unfinished_count)
        except (TypeError, ValueError):
            return

        toast_manager = self.get_toast_manager()
        if unfinished_count <= 0:
            self._close_export_progress_toast()
            return

        message = (
            "正在处理后台任务..."
            if unfinished_count == 1
            else f"正在处理后台任务（{unfinished_count} 个）"
        )
        if message == self._export_status_text:
            return

        self._close_export_progress_toast()
        self._export_status_text = message
        if toast_manager is not None:
            self._export_toast = toast_manager.show_info(message, duration=0)

    @pyqtSlot(object)
    def _on_export_job_finished(self, outcome: ExportOutcome) -> None:
        if not isinstance(outcome, ExportOutcome):
            return

        toast_manager = self.get_toast_manager()
        file_name = os.path.basename(outcome.source_path)
        if outcome.success:
            if outcome.generated_artifact is not None:
                self.model.install_inpaint_artifact(outcome.generated_artifact)
            if not outcome.automatic and toast_manager is not None:
                toast_manager.show_success(
                    f"导出成功\n{outcome.output_path}",
                    5000,
                    outcome.output_path,
                )

            if self._is_same_source_image(
                self.model.get_source_image_path(), outcome.source_path
            ):
                self.resource_manager.release_memory_after_export()
                self.resource_manager.release_image_cache_except_current()
                self._log_memory_snapshot("after-export-cleanup")
            return

        if toast_manager is not None:
            toast_manager.show_error(
                f"{file_name} 导出失败：{outcome.error or '未知错误'}",
                7000,
            )


    @pyqtSlot(object)
    def _apply_inpaint_result(self, result) -> None:
        self.inpaint_service.apply_inpaint_result(result)

    @pyqtSlot(dict)
    def update_multiple_translations(self, translations: dict):
        """Update the translations of several regions as a batch. `translations` is an {index: text} dictionary."""
        if not translations:
            return

        regions = self.model.get_regions()
        old_regions = [copy.deepcopy(region) for region in regions]
        new_regions = [copy.deepcopy(region) for region in regions]
        changed_count = 0

        for raw_index, text in translations.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if not (0 <= index < len(new_regions)):
                continue

            old_region_data = self._merge_live_geometry_state(index, new_regions[index])
            if old_region_data.get("translation", "") == text:
                continue

            new_region_data = self._replace_plain_translation(
                old_region_data,
                translation=text,
                translation_raw=text,
                translation_rich=self._rules_rich_for_full_replacement(
                    old_region_data, text
                ),
            )
            old_regions[index] = copy.deepcopy(old_region_data)
            new_regions[index] = new_region_data
            changed_count += 1

        if not changed_count:
            return

        from .commands import MultiRegionUpdateCommand

        self.execute_command(
            MultiRegionUpdateCommand(
                self.model,
                old_regions,
                new_regions,
                description=f"Batch Update Translations ({changed_count})",
            )
        )

    def _replace_plain_translation(
        self,
        region_data: dict,
        *,
        translation: str,
        translation_raw: str,
        translation_rich: Optional[dict] = None,
    ) -> dict:
        """Write a plain-text translation.

        When ``translation_rich`` is given a synced document, it is written; ``None`` means there is no
        reliable result of moving the styles (a whole-text replacement or a failed sync), and the old rich text is deleted in favour of plain text,
        so the canvas does not go on rendering a stale rich-text body.
        """
        new_region_data = region_data.copy()
        new_region_data["translation"] = translation
        new_region_data["translation_raw"] = translation_raw
        if translation_rich is not None:
            new_region_data["translation_rich"] = translation_rich
        else:
            new_region_data.pop("translation_rich", None)
        return new_region_data

    def _auto_rich_text_rules_enabled(self) -> bool:
        """Switch for applying the rich-text rules automatically while editing (editor menu, stored in the app configuration)."""
        try:
            config = self.config_service.get_config()
        except Exception:
            return True
        return bool(
            getattr(getattr(config, "app", None), "editor_auto_rich_text_rules", True)
        )

    def _sync_rich_for_plain_edit(
        self,
        old_region_data: dict,
        edit_info,
        *,
        raw_mode: bool,
        new_translation: str,
    ) -> Optional[dict]:
        """The alignment logic is in the backend, rich_text_sync; only the fields are picked and forwarded here."""
        from manga_translator.rendering.rich_text_sync import (
            sync_region_rich_translation,
        )

        return sync_region_rich_translation(
            old_region_data.get("translation_rich"),
            edit_info,
            raw_mode=raw_mode,
            new_translation=new_translation,
            direction_value=old_region_data.get("direction", "h"),
            apply_rules=self._auto_rich_text_rules_enabled(),
            old_translation=old_region_data.get("translation", ""),
        )

    def _rules_rich_for_full_replacement(
        self, region_data: dict, translation: str
    ) -> Optional[dict]:
        """Whole-text replacement path: the old rich text is dropped, and the automatic rich-text rules run on the new translation with full semantics."""
        if not self._auto_rich_text_rules_enabled():
            return None
        from manga_translator.rendering.rich_text_sync import (
            sync_region_rich_translation,
        )

        try:
            return sync_region_rich_translation(
                None,
                None,
                raw_mode=False,
                new_translation=translation,
                direction_value=region_data.get("direction", "h"),
                apply_rules=True,
            )
        except Exception as e:
            self.logger.warning(f"auto rich text rules failed: {e}")
            return None

    @pyqtSlot(object)
    def _apply_load_result(self, result: object):
        self.document_service.apply_load_result(result)

    @pyqtSlot()
    def clear_all_masks(self):
        self.inpaint_service.clear_all_masks()

    @pyqtSlot()
    def clear_paint_overlay(self):
        self.inpaint_service.clear_paint_overlay()

    @pyqtSlot()
    def clear_stamp_overlay(self):
        self.inpaint_service.clear_stamp_overlay()

    def _build_region_update_command(
        self,
        *,
        region_index: int,
        old_data: dict,
        new_data: dict,
        description: str,
        merge_key: str,
    ) -> UpdateRegionCommand:
        return UpdateRegionCommand(
            model=self.model,
            region_index=region_index,
            old_data=old_data,
            new_data=new_data,
            description=description,
            merge_key=merge_key,
        )

    @staticmethod
    def _resolve_region_render_params(region_index: int, region_data: dict):
        service = get_render_parameter_service()
        if service is None:
            raise RuntimeError("RenderParameterService is not initialized")
        return service.get_region_parameters(region_index, region_data)

    def _update_region_field(
        self,
        region_index: int,
        field_name: str,
        value,
        *,
        description: str,
        merge_key: str | None = None,
        merge_live_geometry: bool = True,
        current_value=_UNSET,
    ) -> bool:
        old_region_data = self.model.get_region_by_index(region_index)
        if not old_region_data:
            return False

        if merge_live_geometry:
            old_region_data = self._merge_live_geometry_state(
                region_index, old_region_data
            )

        existing_value = (
            old_region_data.get(field_name)
            if current_value is _UNSET
            else current_value
        )
        if existing_value == value:
            return False

        new_region_data = old_region_data.copy()
        new_region_data[field_name] = value

        # Font, translation and similar properties changed -> sync the white box size (anchored on the body centre), so the UI follows the new font size at once.
        if field_name in _FONT_AFFECTING_FIELDS:
            _sync_white_frame_size_for_font_change(
                new_region_data,
                old_region_data,
                self._resolve_region_render_params(region_index, new_region_data),
                self._resolve_region_render_params(region_index, old_region_data),
            )

        command = self._build_region_update_command(
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=description,
            merge_key=merge_key or f"region:{region_index}:{field_name}",
        )
        self.execute_command(command)
        return True

    @staticmethod
    def _normalize_alignment_value(alignment_text: str) -> str:
        raw_text = str(alignment_text or "").strip()
        lower_text = raw_text.lower()
        if lower_text in ("auto", "left", "center", "right"):
            return lower_text

        i18n = get_i18n_manager()
        if i18n:
            localized_map = {
                i18n.translate("alignment_auto"): "auto",
                i18n.translate("alignment_left"): "left",
                i18n.translate("alignment_center"): "center",
                i18n.translate("alignment_right"): "right",
            }
            mapped = localized_map.get(raw_text)
            if mapped is not None:
                return mapped

        fallback_map = {
            "自动": "auto",
            "左对齐": "left",
            "居中": "center",
            "右对齐": "right",
        }
        return fallback_map.get(raw_text, "auto")

    @staticmethod
    def _normalize_direction_value(direction_text: str) -> str:
        raw_text = str(direction_text or "").strip()
        lower_text = raw_text.lower()
        if lower_text in ("v", "vertical"):
            return "vertical"
        if lower_text in ("h", "horizontal"):
            return "horizontal"

        i18n = get_i18n_manager()
        if i18n:
            horizontal_label = i18n.translate("direction_horizontal")
            vertical_label = i18n.translate("direction_vertical")
            if raw_text == vertical_label:
                return "vertical"
            if raw_text == horizontal_label:
                return "horizontal"

        if raw_text in ("竖排",):
            return "vertical"
        if raw_text in ("横排",):
            return "horizontal"
        return "horizontal"

    @pyqtSlot(int, str, object)
    def update_translated_text(self, region_index: int, text: str, edit_info=None):
        # Editing the translation: translation_raw is overwritten too (the rules cannot be reversed, so a plain sync is the only option)
        old_region_data = self.model.get_region_by_index(region_index)
        if not old_region_data:
            return
        new_rich = self._sync_rich_for_plain_edit(
            old_region_data, edit_info, raw_mode=False, new_translation=text
        )
        self._update_translation_pair(
            region_index,
            translation=text,
            translation_raw=text,
            translation_rich=new_rich,
            description=f"Update Translation Region {region_index}",
            merge_key=f"region:{region_index}:translation",
        )

    def _apply_translation_replacements(self, region_data: dict, raw_text: str) -> str:
        """Run the text_replacements rules on a translation; when the rules fail, fall back to the original text."""
        from manga_translator.rendering.text_replacements import apply_replacements

        # Derive direction (see L57: ('h','horizontal','hr') is horizontal, anything else counts as vertical)
        direction_val = region_data.get("direction", "h")
        direction = 0 if direction_val in ("h", "horizontal", "hr") else 1
        try:
            return apply_replacements(raw_text, direction)
        except Exception as e:
            self.logger.warning(f"apply_replacements failed: {e}")
            return raw_text

    @pyqtSlot(int, str, object)
    def update_translation_raw(self, region_index: int, raw_text: str, edit_info=None):
        """Editing the translation before replacement: apply_replacements runs live and syncs to the translation field."""
        old_region_data = self.model.get_region_by_index(region_index)
        if not old_region_data:
            return

        new_translation = self._apply_translation_replacements(
            old_region_data, raw_text
        )
        new_rich = self._sync_rich_for_plain_edit(
            old_region_data, edit_info, raw_mode=True, new_translation=new_translation
        )

        self._update_translation_pair(
            region_index,
            translation=new_translation,
            translation_raw=raw_text,
            translation_rich=new_rich,
            description=f"Update Translation Raw Region {region_index}",
            merge_key=f"region:{region_index}:translation_raw",
        )

    @pyqtSlot(int, object, str)
    def update_translation_rich(
        self, region_index: int, rich_document, plain_text: str
    ):
        old_region_data = self.model.get_region_by_index(region_index)
        if not old_region_data:
            return

        old_region_data = self._merge_live_geometry_state(region_index, old_region_data)
        if (
            old_region_data.get("translation_rich") == rich_document
            and old_region_data.get("translation", "") == plain_text
        ):
            return

        new_region_data = old_region_data.copy()
        text_changed = old_region_data.get("translation", "") != plain_text
        new_region_data["translation"] = plain_text
        if text_changed or "translation_raw" not in old_region_data:
            # After the rich-text body changes, the translation before replacement cannot be derived reliably; a pure style change keeps the original raw.
            new_region_data["translation_raw"] = plain_text
        new_region_data["translation_rich"] = rich_document

        _sync_white_frame_size_for_font_change(
            new_region_data,
            old_region_data,
            self._resolve_region_render_params(region_index, new_region_data),
            self._resolve_region_render_params(region_index, old_region_data),
        )

        command = self._build_region_update_command(
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Update Rich Translation Region {region_index}",
            merge_key=f"region:{region_index}:translation_rich",
        )
        self.execute_command(command)

    def _update_translation_pair(
        self,
        region_index: int,
        *,
        translation: str,
        translation_raw: str,
        description: str,
        merge_key: str,
        translation_rich: Optional[dict] = None,
    ) -> bool:
        """Update translation and translation_raw together in one undo command (an undo rolls both back)."""
        old_region_data = self.model.get_region_by_index(region_index)
        if not old_region_data:
            return False

        old_region_data = self._merge_live_geometry_state(region_index, old_region_data)

        if (
            old_region_data.get("translation", "") == translation
            and old_region_data.get("translation_raw", "") == translation_raw
        ):
            return False

        new_region_data = self._replace_plain_translation(
            old_region_data,
            translation=translation,
            translation_raw=translation_raw,
            translation_rich=translation_rich,
        )

        # translation is a member of _FONT_AFFECTING_FIELDS, so the white box size is synced after a change
        _sync_white_frame_size_for_font_change(
            new_region_data,
            old_region_data,
            self._resolve_region_render_params(region_index, new_region_data),
            self._resolve_region_render_params(region_index, old_region_data),
        )

        command = self._build_region_update_command(
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=description,
            merge_key=merge_key,
        )
        self.execute_command(command)
        return True

    @pyqtSlot(int, str)
    def update_original_text(self, region_index: int, text: str):
        self._update_region_field(
            region_index,
            "text",
            text,
            description=f"Update Original Text Region {region_index}",
        )

    @pyqtSlot(int, int)
    def update_font_size(self, region_index: int, size: int):
        self._update_region_field(
            region_index,
            "font_size",
            size,
            description=f"Update Font Size Region {region_index}",
        )

    @pyqtSlot(int, str)
    def update_font_color(self, region_index: int, color: str):
        self._update_region_field(
            region_index,
            "font_color",
            color,
            description=f"Update Font Color Region {region_index}",
        )

    @pyqtSlot(int, str)
    def update_stroke_color(self, region_index: int, hex_color: str):
        from PyQt6.QtGui import QColor

        c = QColor(hex_color)
        new_bg_colors = [c.red(), c.green(), c.blue()]
        self._update_region_field(
            region_index,
            "bg_colors",
            new_bg_colors,
            description=f"Update Stroke Color Region {region_index}",
        )

    @pyqtSlot(int, float)
    def update_stroke_width(self, region_index: int, value: float):
        self._update_region_field(
            region_index,
            "stroke_width",
            value,
            description=f"Update Stroke Width Region {region_index}",
        )

    @pyqtSlot(int, float)
    def update_line_spacing(self, region_index: int, value: float):
        old_region_data = self.model.get_region_by_index(region_index)
        if not old_region_data:
            return
        current_value = old_region_data.get("line_spacing")
        if current_value is None:
            current_value = self.config_service.get_config().render.line_spacing or 1.0
        self._update_region_field(
            region_index,
            "line_spacing",
            value,
            description=f"Update Line Spacing Region {region_index}",
            current_value=current_value,
        )

    @pyqtSlot(int, float)
    def update_letter_spacing(self, region_index: int, value: float):
        old_region_data = self.model.get_region_by_index(region_index)
        if not old_region_data:
            return
        current_value = old_region_data.get("letter_spacing")
        if current_value is None:
            current_value = (
                self.config_service.get_config().render.letter_spacing or 1.0
            )
        self._update_region_field(
            region_index,
            "letter_spacing",
            value,
            description=f"Update Letter Spacing Region {region_index}",
            current_value=current_value,
        )

    @staticmethod
    def _build_rotated_region_data(
        old_region_data: dict, value: float
    ) -> Optional[dict]:
        target_angle = float(value)
        current_angle = float(old_region_data.get("angle", 0.0) or 0.0)
        if np.isclose(current_angle, target_angle, atol=1e-6):
            return None

        geo = RegionGeometryState.from_region_data(old_region_data)
        wf_local = geo.white_frame_local
        if wf_local is not None and len(wf_local) == 4:
            left, top, right, bottom = wf_local
            pivot_lx = (left + right) / 2.0
            pivot_ly = (top + bottom) / 2.0
        else:
            pivot_lx = pivot_ly = 0.0

        pivot_scene_x, pivot_scene_y = geo.local_to_world(pivot_lx, pivot_ly)
        theta = np.radians(target_angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        new_cx = pivot_scene_x - (pivot_lx * cos_t - pivot_ly * sin_t)
        new_cy = pivot_scene_y - (pivot_lx * sin_t + pivot_ly * cos_t)

        old_center = geo.center if len(geo.center) >= 2 else [new_cx, new_cy]
        delta_x = float(new_cx) - float(old_center[0])
        delta_y = float(new_cy) - float(old_center[1])
        new_lines = []
        for poly in old_region_data.get("lines", []):
            new_poly = []
            for point in poly:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    new_poly.append(
                        [float(point[0]) + delta_x, float(point[1]) + delta_y]
                    )
            if new_poly:
                new_lines.append(new_poly)

        return build_rotate_region_data(
            old_region_data,
            target_angle,
            new_center=[new_cx, new_cy],
            new_lines=new_lines or None,
        )

    @pyqtSlot(int, float)
    def update_angle(self, region_index: int, value: float):
        old_region_data = self.model.get_region_by_index(region_index)
        if not old_region_data:
            return

        old_region_data = self._merge_live_geometry_state(region_index, old_region_data)
        new_region_data = self._build_rotated_region_data(old_region_data, value)
        if new_region_data is None:
            return
        command = self._build_region_update_command(
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Update Rotation Region {region_index}",
            merge_key=f"region:{region_index}:angle",
        )
        self.execute_command(command)

    @pyqtSlot(int, str)
    def update_font_family(self, region_index: int, font_family: str):
        self._update_region_field(
            region_index,
            "font_family",
            font_family,
            description=f"Update Font Family Region {region_index}",
        )

    @pyqtSlot(int, str)
    def update_alignment(self, region_index: int, alignment_text: str):
        alignment_value = self._normalize_alignment_value(alignment_text)
        self._update_region_field(
            region_index,
            "alignment",
            alignment_value,
            description=f"Update Alignment to {alignment_value}",
        )

    @pyqtSlot(int, dict)
    def update_region_geometry(self, region_index: int, new_region_data: dict):
        """Handle a geometry change of a region coming from the view."""
        # RegionTextItem no longer modifies self.region_data before calling the callback,
        # so the correct old data can be read from the model
        old_region_data = self.model.get_region_by_index(region_index)
        if not old_region_data:
            return

        # Deep copy, to avoid reference problems
        old_region_data = copy.deepcopy(old_region_data)

        command = UpdateRegionCommand(
            model=self.model,
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Resize/Move/Rotate Region {region_index}",
        )
        self.execute_command(command)

    # ------------------------------------------------------------------
    # Align and distribute
    # ------------------------------------------------------------------

    def align_regions(self, mode: str, reference: str) -> None:
        """Align the selected regions as a batch.

        mode: top / vertical_center / bottom / left / horizontal_center / right
        reference: "selection" | "canvas"
        """
        from .alignment_service import align_items

        view = self.get_graphics_view()
        if view is None:
            return
        items = [item for item in view._region_items if item.isSelected()]
        if not items:
            return

        canvas_rect = None
        if reference == "canvas":
            r = view.get_image_scene_rect()
            if r is not None:
                canvas_rect = (r.left(), r.top(), r.right(), r.bottom())

        results = align_items(items, mode, reference, canvas_rect)
        if not results:
            return

        # Build one batch command: change the model centre and move the white box and text of the item with it
        regions = self.model.get_regions()
        old_regions = [dict(r) for r in regions]
        new_regions = [dict(r) for r in regions]
        for idx, new_cx, new_cy in results:
            new_regions[idx]["center"] = [new_cx, new_cy]

        from .commands import MultiRegionUpdateCommand

        mode_names = {
            "top": "Top Align",
            "vertical_center": "Vertical Center",
            "bottom": "Bottom Align",
            "left": "Left Align",
            "horizontal_center": "Horizontal Center",
            "right": "Right Align",
        }
        cmd = MultiRegionUpdateCommand(
            self.model,
            old_regions,
            new_regions,
            description=f"{mode_names.get(mode, 'Align')} ({reference})",
        )
        self.execute_command(cmd)

        # Does not rely on debounce -> asynchronous rebuild; moves the white box and text of the item at once, as dragging does
        self._sync_items_positions(results, items)

    def _sync_items_positions(self, results, items):
        """After aligning, sync item.center to the new position at once (only center moves, not wf_local).

        The white box does not move relative to center in local coordinates; only center changes, which moves the whole item to the target position.
        The model center was already updated by MultiRegionUpdateCommand; only the look of the Qt item is refreshed here.
        """
        from PyQt6.QtCore import QPointF

        for idx, new_cx, new_cy in results:
            item = None
            for it in items:
                if it.region_index == idx:
                    item = it
                    break
            if item is None or item.scene() is None:
                continue

            old_pos = item.pos()
            dx = new_cx - float(old_pos.x())
            dy = new_cy - float(old_pos.y())
            if abs(dx) < 0.01 and abs(dy) < 0.01:
                continue

            old_rect = item.sceneBoundingRect()
            item.prepareGeometryChange()
            item._shape_path = None
            item.geo.center = [new_cx, new_cy]
            item.visual_center = QPointF(new_cx, new_cy)
            item.setPos(new_cx, new_cy)
            item.update()
            item._invalidate_scene_rect(old_rect)

    def distribute_regions(self, mode: str) -> None:
        """Distribute the selected regions evenly as a batch.

        mode: top / vertical_center / bottom / left / horizontal_center / right
        """
        view = self.get_graphics_view()
        if view is None:
            return
        items = [item for item in view._region_items if item.isSelected()]

        # Distribute by spacing vs by edges
        if mode in ("spacing_v", "spacing_h"):
            if len(items) < 3:
                return
            from .alignment_service import distribute_spacing_items

            orientation = "vertical" if mode == "spacing_v" else "horizontal"
            results = distribute_spacing_items(items, orientation)
            desc = (
                "Distribute Spacing V"
                if mode == "spacing_v"
                else "Distribute Spacing H"
            )
        else:
            from .alignment_service import distribute_items

            if len(items) < 3:
                return
            results = distribute_items(items, mode)
            mode_names = {
                "top": "Top Distribute",
                "vertical_center": "Vertical Center Distribute",
                "bottom": "Bottom Distribute",
                "left": "Left Distribute",
                "horizontal_center": "Horizontal Center Distribute",
                "right": "Right Distribute",
            }
            desc = mode_names.get(mode, "Distribute")

        if not results:
            return

        regions = self.model.get_regions()
        old_regions = [dict(r) for r in regions]
        new_regions = [dict(r) for r in regions]
        for idx, new_cx, new_cy in results:
            new_regions[idx]["center"] = [new_cx, new_cy]

        from .commands import MultiRegionUpdateCommand

        cmd = MultiRegionUpdateCommand(
            self.model, old_regions, new_regions, description=desc
        )
        self.execute_command(cmd)

        self._sync_items_positions(results, items)

    @pyqtSlot(int, str)
    def update_direction(self, region_index: int, direction_text: str):
        direction_value = self._normalize_direction_value(direction_text)
        self._update_region_field(
            region_index,
            "direction",
            direction_value,
            description=f"Update Direction to {direction_value}",
        )

    @pyqtSlot(list, dict)
    def update_region_style_patch(self, region_indices: list, patch: dict) -> None:
        """Apply one style patch to a selection as one undo command/model notification."""
        if not isinstance(patch, dict):
            return

        normalized_patch = {}
        for key, value in patch.items():
            if key not in _STYLE_PATCH_FIELDS:
                continue
            try:
                if key == "font_size":
                    normalized_patch[key] = max(1, int(value))
                elif key in {"stroke_width", "line_spacing", "letter_spacing", "angle"}:
                    normalized_patch[key] = float(value)
                elif key == "alignment":
                    normalized_patch[key] = self._normalize_alignment_value(value)
                elif key == "direction":
                    normalized_patch[key] = self._normalize_direction_value(value)
                else:
                    normalized_patch[key] = str(value or "")
            except (TypeError, ValueError):
                continue
        if not normalized_patch:
            return

        regions = self.model.get_regions()
        indices = set()
        for raw_index in region_indices or []:
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(regions):
                indices.add(index)
        indices = sorted(indices)
        if not indices:
            return

        from PyQt6.QtGui import QColor

        from .commands import MultiRegionUpdateCommand

        old_regions = list(regions)
        new_regions = list(regions)
        changed_fields = set()
        changed_count = 0
        render_defaults = self.config_service.get_config().render

        for index in indices:
            old_region_data = self._merge_live_geometry_state(index, regions[index])
            new_region_data = old_region_data.copy()
            region_changed = False
            font_metrics_changed = False

            if "angle" in normalized_patch:
                rotated = self._build_rotated_region_data(
                    old_region_data, normalized_patch["angle"]
                )
                if rotated is not None:
                    new_region_data = rotated
                    region_changed = True
                    changed_fields.add("angle")

            for field_name, value in normalized_patch.items():
                if field_name == "angle":
                    continue

                stored_field = field_name
                stored_value = value
                if field_name == "stroke_color":
                    color = QColor(value)
                    if not color.isValid():
                        continue
                    stored_field = "bg_colors"
                    stored_value = [color.red(), color.green(), color.blue()]

                current_value = new_region_data.get(stored_field)
                if (
                    field_name in {"line_spacing", "letter_spacing"}
                    and current_value is None
                ):
                    current_value = getattr(render_defaults, field_name, None) or 1.0
                try:
                    unchanged = (
                        np.isclose(float(current_value), float(stored_value), atol=1e-9)
                        if field_name
                        in {"stroke_width", "line_spacing", "letter_spacing"}
                        and current_value is not None
                        else current_value == stored_value
                    )
                except (TypeError, ValueError):
                    unchanged = current_value == stored_value
                if unchanged:
                    continue

                new_region_data[stored_field] = copy.deepcopy(stored_value)
                region_changed = True
                changed_fields.add(stored_field)
                if stored_field in _FONT_AFFECTING_FIELDS:
                    font_metrics_changed = True

            if not region_changed:
                continue
            if font_metrics_changed:
                _sync_white_frame_size_for_font_change(
                    new_region_data,
                    old_region_data,
                    self._resolve_region_render_params(index, new_region_data),
                    self._resolve_region_render_params(index, old_region_data),
                )
            old_regions[index] = copy.deepcopy(old_region_data)
            new_regions[index] = new_region_data
            changed_count += 1

        if not changed_count:
            return
        command = MultiRegionUpdateCommand(
            self.model,
            old_regions,
            new_regions,
            description=f"Update Region Style ({changed_count})",
            fields=sorted(changed_fields),
            source="property-panel",
        )
        if command.has_changes():
            self.execute_command(command)

    def _execute_command_batch(self, commands: list, macro_name: str) -> None:
        with self.history_service.macro(macro_name):
            for command in commands:
                self.execute_command(command, update_ui=False)
        self._update_undo_redo_buttons()

    def execute_command(self, command, update_ui: bool = True):
        """Run a command and update the UI - with Qt's QUndoStack"""
        if command is None:
            return
        self.history_service.execute(command)
        if update_ui:
            self._update_undo_redo_buttons()

    def undo(self):
        """Undo - with Qt's QUndoStack"""
        self.history_service.undo()
        self._update_undo_redo_buttons()

    def redo(self):
        """Redo - with Qt's QUndoStack"""
        self.history_service.redo()
        self._update_undo_redo_buttons()

    # --- Paste overlay operations ---
    def _normalize_paste_overlays(self, overlays) -> list:
        """Normalise the paste overlay list (defaults and ids filled in), so the ids of undo/redo snapshots are stable."""
        from editor.paste_overlay_state import serialize_paste_overlays

        return serialize_paste_overlays(overlays or [])

    def add_paste_overlay(self, overlay: dict) -> bool:
        """Add a paste overlay (with undo). overlay may be a dictionary that is not normalised yet."""
        from editor.commands import PasteOverlaysReplaceCommand

        if self.model.get_source_image_path() is None:
            return False
        before = self.model.get_paste_overlays()
        after = self._normalize_paste_overlays(before + [overlay])
        if after == before:
            return False
        self.execute_command(
            PasteOverlaysReplaceCommand(
                self.model, before, after, description="Add Paste Overlay"
            )
        )
        return True

    def update_paste_overlay(self, overlay_id: str, patch: dict) -> bool:
        """Update a paste overlay by id (patch is merged field by field; with undo)."""
        from editor.commands import PasteOverlaysReplaceCommand

        if self.model.get_source_image_path() is None:
            return False
        before = self.model.get_paste_overlays()
        after = copy.deepcopy(before)
        for item in after:
            if item.get("id") == overlay_id:
                item.update(copy.deepcopy(patch))
                break
        else:
            return False
        after = self._normalize_paste_overlays(after)
        if after == before:
            return False
        self.execute_command(
            PasteOverlaysReplaceCommand(
                self.model, before, after, description="Update Paste Overlay"
            )
        )
        return True

    def remove_paste_overlay(self, overlay_id: str) -> bool:
        """Delete a paste overlay by id (with undo)."""
        from editor.commands import PasteOverlaysReplaceCommand

        if self.model.get_source_image_path() is None:
            return False
        before = self.model.get_paste_overlays()
        after = [item for item in before if item.get("id") != overlay_id]
        if len(after) == len(before):
            return False
        self.execute_command(
            PasteOverlaysReplaceCommand(
                self.model, before, after, description="Remove Paste Overlay"
            )
        )
        return True

    def replace_paste_overlays(self, overlays: list) -> bool:
        """Replace the whole paste overlay list (for drag-and-drop import, z-order changes and so on; with undo)."""
        from editor.commands import PasteOverlaysReplaceCommand

        if self.model.get_source_image_path() is None:
            return False
        before = self.model.get_paste_overlays()
        after = self._normalize_paste_overlays(overlays)
        if after == before:
            return False
        self.execute_command(
            PasteOverlaysReplaceCommand(
                self.model, before, after, description="Replace Paste Overlays"
            )
        )
        return True

    def copy_paste_overlay(self, overlay_id: str) -> bool:
        """Copy a paste overlay to the overlay clipboard."""
        if self.model.get_source_image_path() is None:
            return False
        for overlay in self.model.get_paste_overlays():
            if overlay.get("id") == overlay_id:
                self._paste_overlay_clipboard = copy.deepcopy(overlay)
                self._last_clipboard_kind = "paste_overlay"
                return True
        return False

    def last_clipboard_kind(self) -> str | None:
        """Return the type of the object copied most recently ('paste_overlay' | 'region' | None)."""
        return getattr(self, "_last_clipboard_kind", None)

    def paste_overlay_clipboard_available(self) -> bool:
        return bool(getattr(self, "_paste_overlay_clipboard", None))

    def paste_paste_overlay(self, center: tuple[float, float] | None = None) -> bool:
        """Paste a paste overlay: without center it is offset by (+20,+20) from the original position; with center it lands there."""
        clipboard = getattr(self, "_paste_overlay_clipboard", None)
        if clipboard is None or self.model.get_source_image_path() is None:
            return False
        clone = copy.deepcopy(clipboard)
        clone.pop("id", None)  # Generate a new id, so it does not repeat the source
        if center is not None:
            center_x = center.x() if hasattr(center, "x") else center[0]
            center_y = center.y() if hasattr(center, "y") else center[1]
            clone["center_x"], clone["center_y"] = float(center_x), float(center_y)
        else:
            clone["center_x"] = float(clone.get("center_x", 0.0)) + 20.0
            clone["center_y"] = float(clone.get("center_y", 0.0)) + 20.0
        return self.add_paste_overlay(clone)

    def duplicate_paste_overlay(self, overlay_id: str) -> bool:
        """Duplicate a paste overlay in place (offset +20, +20), for the shortcut and the context menu."""
        if not self.copy_paste_overlay(overlay_id):
            return False
        return self.paste_paste_overlay()

    # --- Context menu methods ---
    def ocr_regions(self, region_indices: list):
        """Run OCR on the given regions, with the same logic as the UI button"""
        if not region_indices:
            return

        # Keep the current selection for now
        original_selection = self.model.get_selection()

        # Set the selection to the regions to run OCR on
        self.model.set_selection(region_indices)

        # Call the existing OCR method (it uses the OCR model set in the UI)
        self.run_ocr_for_selection()

        # Restore the original selection
        self.model.set_selection(original_selection)

    def translate_regions(self, region_indices: list):
        """Translate the text of the given regions, with the same logic as the UI button"""
        if not region_indices:
            return

        # Keep the current selection for now
        original_selection = self.model.get_selection()

        # Set the selection to the regions to translate
        self.model.set_selection(region_indices)

        # Call the existing translation method (it uses the translator and target language set in the UI)
        self.run_translation_for_selection()

        # Restore the original selection
        self.model.set_selection(original_selection)

    def copy_region(self, region_index: int):
        """Copy the data of the given region"""
        self.copy_regions([region_index])

    def copy_regions(self, region_indices: list):
        """Copy the data of several regions"""
        regions = self.model.get_regions()
        region_data = [
            copy.deepcopy(regions[index])
            for index in region_indices
            if isinstance(index, int) and 0 <= index < len(regions)
        ]
        if not region_data:
            self.logger.error("No valid region to copy")
            return

        self.history_service.copy_to_clipboard(region_data)
        self._last_clipboard_kind = "region"

    def paste_region_style(self, region_index: int):
        """Paste the copied style onto the given region"""
        clipboard_data = self.history_service.paste_from_clipboard()
        if isinstance(clipboard_data, list):
            clipboard_data = clipboard_data[0] if clipboard_data else None
        if not clipboard_data:
            self.logger.warning("No copied region data available")
            return

        region_data = self.model.get_region_by_index(region_index)
        if not region_data:
            self.logger.error(f"Region {region_index} does not exist")
            return

        # Copy the style properties, but keep the position and the text
        old_region_data = region_data.copy()
        new_region_data = region_data.copy()

        # Copy the style properties
        style_keys = [
            "font_family",
            "font_size",
            "font_color",
            "alignment",
            "direction",
            "line_spacing",
            "letter_spacing",
        ]
        for key in style_keys:
            if key in clipboard_data:
                new_region_data[key] = clipboard_data[key]

        command = UpdateRegionCommand(
            model=self.model,
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Paste Style to Region {region_index}",
        )
        self.execute_command(command)

    def delete_regions(self, region_indices: list):
        """Delete the given regions."""
        if not region_indices:
            return

        graphics_view = self.get_graphics_view()
        if graphics_view is not None:
            graphics_view.clear_pending_geometry_edits()

        from editor.commands import DeleteRegionCommand

        regions = self.model.get_regions()
        recover_removed_text = self._delete_and_recover_enabled()
        current_raw_mask = (
            self._copy_mask(self.model.get_raw_mask()) if recover_removed_text else None
        )
        current_refined_mask = (
            self._copy_mask(self.model.get_refined_mask())
            if recover_removed_text
            else None
        )
        pending_commands = []
        sorted_indices = sorted(
            {
                int(region_index)
                for region_index in region_indices
                if isinstance(region_index, int)
            },
            reverse=True,
        )

        for region_index in sorted_indices:
            if 0 <= region_index < len(regions):
                old_raw_mask = new_raw_mask = old_refined_mask = new_refined_mask = (
                    _NO_MASK_CHANGE
                )
                if recover_removed_text:
                    from editor.mask_region import remove_region_from_mask

                    if current_raw_mask is not None:
                        old_raw_mask = current_raw_mask
                        new_raw_mask = remove_region_from_mask(
                            current_raw_mask,
                            regions[region_index],
                            self._delete_mask_expand_px(),
                        )
                        current_raw_mask = new_raw_mask
                    if current_refined_mask is not None:
                        old_refined_mask = current_refined_mask
                        new_refined_mask = remove_region_from_mask(
                            current_refined_mask,
                            regions[region_index],
                            self._delete_mask_expand_px(),
                        )
                        current_refined_mask = new_refined_mask
                pending_commands.append(
                    DeleteRegionCommand(
                        model=self.model,
                        region_index=region_index,
                        region_data=regions[region_index],
                        description=f"Delete Region {region_index}",
                        old_raw_mask=old_raw_mask,
                        new_raw_mask=new_raw_mask,
                        old_refined_mask=old_refined_mask,
                        new_refined_mask=new_refined_mask,
                    )
                )

        if pending_commands:
            self._execute_command_batch(
                pending_commands,
                f"Delete Regions ({len(pending_commands)} ops)",
            )

        # Clear the selection
        self.model.set_selection([])

    @staticmethod
    def _copy_mask(mask):
        return None if mask is None else np.array(mask, copy=True)

    def _delete_and_recover_enabled(self) -> bool:
        try:
            config = self.config_service.get_config()
            return bool(
                getattr(
                    getattr(config, "app", None), "editor_delete_and_recover", False
                )
            )
        except Exception:
            return False

    def _delete_mask_expand_px(self) -> int:
        """Match the approximate dilation radius used by mask refinement."""
        try:
            offset = int(
                getattr(self.config_service.get_config(), "mask_dilation_offset", 0)
            )
        except (TypeError, ValueError, AttributeError):
            offset = 0
        return max(0, int(round(offset * 0.3)))

    def enter_drawing_mode(self):
        """Enter drawing mode to add a new text box"""
        # Clear the current selection
        self.model.set_selection([])

        # Set the tool to drawing a text box
        self.model.set_active_tool("draw_textbox")

    @staticmethod
    def _region_center(region_data: dict):
        if "center" in region_data:
            return float(region_data["center"][0]), float(region_data["center"][1])
        points = [point for line in region_data.get("lines") or [] for point in line]
        if not points:
            return 0.0, 0.0
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    def paste_region(self, mouse_pos=None):
        """Paste the copied region at a new position"""
        self.paste_regions(mouse_pos)

    def paste_regions(self, mouse_pos=None):
        """Paste all regions in the clipboard, keeping their positions relative to each other

        Args:
            mouse_pos: the mouse position (scene coordinates); when given, the whole group is pasted there
        """
        clipboard_data = self.history_service.paste_from_clipboard()
        if not clipboard_data:
            self.logger.warning("No copied region data available")
            return

        if not isinstance(clipboard_data, list):
            clipboard_data = [clipboard_data]

        centers = [self._region_center(item) for item in clipboard_data]
        center_x = sum(item[0] for item in centers) / len(centers)
        center_y = sum(item[1] for item in centers) / len(centers)

        if mouse_pos:
            new_center_x, new_center_y = mouse_pos.x(), mouse_pos.y()
        else:
            new_center_x = center_x + 20
            new_center_y = center_y + 20

        offset_x = new_center_x - center_x
        offset_y = new_center_y - center_y

        from editor.commands import AddRegionCommand

        moved_points = set()
        commands = []
        for region_data in clipboard_data:
            if "center" in region_data:
                region_data["center"] = [
                    region_data["center"][0] + offset_x,
                    region_data["center"][1] + offset_y,
                ]
            for key in ("lines", "polygons"):
                for polygon in region_data.get(key) or []:
                    for point in polygon:
                        if id(point) in moved_points:
                            continue
                        moved_points.add(id(point))
                        point[0] += offset_x
                        point[1] += offset_y
            commands.append(
                AddRegionCommand(
                    model=self.model, region_data=region_data, description="Paste Region"
                )
            )

        self._execute_command_batch(commands, f"Paste Regions ({len(commands)} ops)")

        total = len(self.model.get_regions())
        self.model.set_selection(list(range(total - len(commands), total)))

    @pyqtSlot(bool, bool)
    def _on_history_undo_redo_state_changed(self, can_undo: bool, can_redo: bool):
        """Callback for a change of the history stack state."""
        toolbar = self.get_toolbar()
        if toolbar is not None:
            toolbar.update_undo_redo_state(can_undo, can_redo)

    def _update_undo_redo_buttons(self):
        """Refresh the state of the undo/redo buttons on request."""
        # Check whether history_service is initialised
        if not hasattr(self, "history_service") or self.history_service is None:
            return

        can_undo = self.history_service.can_undo()
        can_redo = self.history_service.can_redo()
        self._on_history_undo_redo_state_changed(can_undo, can_redo)

    @pyqtSlot()
    def save_editor_state(self) -> bool:
        """Save the editor project data, without producing an exported image."""
        return self.export_service.save_editor_state()

    @pyqtSlot()
    def export_image(
        self,
        automatic: bool = False,
    ):
        self.commit_pending_edits()
        return self.export_service.export_image(automatic=automatic)

    @pyqtSlot(str)
    def set_display_mode(self, mode: str):
        """Set the display mode of the editor."""
        compare_enabled = mode == "compare_original_split"
        region_mode = "full" if compare_enabled else mode
        if region_mode not in {"full", "text_only", "box_only", "none"}:
            region_mode = "full"

        self.logger.info(
            f"Toolbar: Display mode changed to '{mode}' (region mode: '{region_mode}', compare={compare_enabled})."
        )
        self.set_compare_mode(compare_enabled)
        self.model.set_region_display_mode(region_mode)

    @pyqtSlot(int)
    def set_original_image_alpha(self, alpha: int):
        """Record the toolbar percentage as the user's opacity for this session."""
        self.model.set_original_image_alpha_override(alpha / 100.0)

    def handle_global_render_setting_change(self):
        """Forces a re-render of all regions when a global render setting has changed."""

        # Clear the parameter service cache to ensure new global defaults are used
        from services import get_render_parameter_service

        render_parameter_service = get_render_parameter_service()
        render_parameter_service.clear_cache()

        # Global render parameters affect the derived render result of every region; only the view cache is rebuilt, not the regions or their ids.
        self.model.refresh_regions()

    @pyqtSlot()
    def run_ocr_for_selection(self):
        selected_indices = self.model.get_selection()
        if not selected_indices:
            return
        image = self.model.get_image()
        if not image:
            return

        all_regions = self.model.get_regions()
        valid_indices = [
            index for index in selected_indices if 0 <= index < len(all_regions)
        ]
        if not valid_indices:
            return

        selected_regions_data = [copy.deepcopy(all_regions[i]) for i in valid_indices]
        region_ids = [self.model.get_region_id(i) for i in valid_indices]
        ocr_config = None
        property_panel = self.get_property_panel()
        if property_panel is not None:
            selected_ocr = property_panel.get_selected_ocr_model()
            if selected_ocr:
                from manga_translator.config import Ocr, OcrConfig

                full_config = self.config_service.get_config()
                current_ocr_config = (
                    full_config.ocr if hasattr(full_config, "ocr") else OcrConfig()
                )
                try:
                    ocr_payload = (
                        current_ocr_config.model_dump()
                        if hasattr(current_ocr_config, "model_dump")
                        else {}
                    )
                    ocr_payload["ocr"] = Ocr(selected_ocr)
                    ocr_config = OcrConfig(**ocr_payload)
                    self.logger.info(
                        f"Using OCR model from property panel: {selected_ocr}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Invalid OCR selection '{selected_ocr}', using default: {e}"
                    )

        # Show the start toast and keep a reference so it can be closed later
        self._ocr_toast = None
        toast_manager = self.get_toast_manager()
        if toast_manager is not None:
            self._ocr_toast = toast_manager.show_info("正在识别...", duration=0)

        self.async_service.submit_task(
            self._async_ocr_task(image, selected_regions_data, region_ids, ocr_config)
        )

    @pyqtSlot(object)
    def on_regions_update_finished(self, request):
        """Asynchronous write-back of OCR and translation: the main thread locates the regions again by region_id and notifies the view once."""
        applied_count = 0
        skipped_count = 0
        try:
            if not isinstance(request, _AsyncRegionUpdateRequest):
                return

            if not request.field_name:
                skipped_count = len(request.updates)
                return

            applied: dict[int, dict] = {}
            for region_id, value in request.updates:
                index = (
                    self.model.find_region_index(region_id)
                    if region_id is not None
                    else None
                )
                region_data = (
                    self.model.get_region_by_index(index) if index is not None else None
                )
                if index is None or region_data is None:
                    skipped_count += 1
                    continue
                current_region_data = applied.get(index, region_data)
                if request.field_name == "translation":
                    # The same path as a manual edit: the translation goes through the replacement rules first, raw keeps the original translation
                    replaced_value = self._apply_translation_replacements(
                        current_region_data, value
                    )
                    new_region_data = self._replace_plain_translation(
                        current_region_data,
                        translation=replaced_value,
                        translation_raw=value,
                        translation_rich=self._rules_rich_for_full_replacement(
                            current_region_data, replaced_value
                        ),
                    )
                else:
                    new_region_data = current_region_data.copy()
                    new_region_data[request.field_name] = value
                applied[index] = new_region_data

            if applied:
                # Goes through the undo stack: OCR and translation results can be undone with Ctrl+Z, and the "unsaved" check on switching images sees them
                from .commands import MultiRegionUpdateCommand

                fields = (
                    ["translation", "translation_raw", "translation_rich"]
                    if request.field_name == "translation"
                    else [request.field_name]
                )
                old_regions = self.model.get_regions()
                new_regions = list(old_regions)
                for index, new_region_data in applied.items():
                    new_regions[index] = new_region_data
                description = (
                    "OCR Update" if request.task_kind == "ocr" else "Translation Update"
                )
                command = MultiRegionUpdateCommand(
                    self.model,
                    old_regions,
                    new_regions,
                    description=description,
                    fields=fields,
                    source="async",
                )
                if command.has_changes():
                    self.execute_command(command)
                applied_count = len(applied)
        except Exception as exc:
            self.logger.error(
                "Failed to apply async region updates: %s", exc, exc_info=True
            )
            skipped_count = (
                len(request.updates)
                if isinstance(request, _AsyncRegionUpdateRequest)
                else 0
            )
        finally:
            if isinstance(request, _AsyncRegionUpdateRequest):
                self._finish_async_region_update(
                    request.task_kind,
                    applied_count=applied_count,
                    skipped_count=skipped_count,
                    error_count=request.error_count,
                )

    @pyqtSlot(str, str)
    def _on_ocr_finished(self, status: str, message: str):
        """Handle the toast on the main thread after OCR is done."""
        self._finalize_progress_toast("_ocr_toast", status, message)

    @pyqtSlot(str, str)
    def _on_translation_finished(self, status: str, message: str):
        """Handle the toast on the main thread after translation is done."""
        self._finalize_progress_toast("_translation_toast", status, message)

    async def _async_ocr_task(self, image, regions_to_process, region_ids, ocr_config):
        pending_updates: list[tuple[int, str]] = []
        error_count = 0
        for i, region_data in enumerate(regions_to_process):
            try:
                ocr_result = await self.ocr_service.recognize_region(
                    image, region_data, config=ocr_config
                )
                if ocr_result and ocr_result.text:
                    pending_updates.append((region_ids[i], ocr_result.text))
            except Exception as e:
                self.logger.error(f"OCR failed: {e}")
                error_count += 1

        if pending_updates:
            self._queue_async_region_updates(
                pending_updates,
                field_name="text",
                task_kind="ocr",
                error_count=error_count,
            )
            return

        if error_count > 0:
            self._ocr_finished.emit("error", "识别失败")
            return
        self._ocr_finished.emit("warning", "未识别到可更新的文本")

    @pyqtSlot()
    def run_translation_for_selection(self):
        selected_indices = self.model.get_selection()
        if not selected_indices:
            return
        image = self.model.get_image()
        if not image:
            return

        all_regions = self.model.get_regions()
        valid_indices = [
            index for index in selected_indices if 0 <= index < len(all_regions)
        ]
        if not valid_indices:
            return

        selected_regions_data = [copy.deepcopy(all_regions[i]) for i in valid_indices]
        texts_to_translate = [r.get("text", "") for r in selected_regions_data]
        region_ids = [self.model.get_region_id(i) for i in valid_indices]
        regions_context = copy.deepcopy(all_regions)
        translator_to_use = None
        target_lang_to_use = None

        property_panel = self.get_property_panel()
        if property_panel is not None:
            selected_translator = property_panel.get_selected_translator()
            selected_target_lang = property_panel.get_selected_target_language()

            if selected_translator:
                from manga_translator.config import Translator

                try:
                    translator_to_use = Translator(selected_translator)
                    self.logger.info(
                        f"Using translator from property panel: {selected_translator}"
                    )
                except (ValueError, AttributeError) as e:
                    self.logger.warning(
                        f"Invalid translator selection '{selected_translator}', using default: {e}"
                    )

            if selected_target_lang:
                target_lang_to_use = selected_target_lang
                self.logger.info(
                    f"Using target language from property panel: {selected_target_lang}"
                )

        # Show the start toast and keep a reference so it can be closed later
        self._translation_toast = None
        toast_manager = self.get_toast_manager()
        if toast_manager is not None:
            self._translation_toast = toast_manager.show_info("正在翻译...", duration=0)

        # Pass all regions to give context, but only translate the selected text
        self.async_service.submit_task(
            self._async_translation_task(
                texts_to_translate,
                region_ids,
                image,
                regions_context,
                translator_to_use,
                target_lang_to_use,
                load_chapter_context(self.model.get_source_image_path(), self.config_service.root_dir),
            )
        )

    async def _async_translation_task(
        self,
        texts,
        region_ids,
        image,
        regions,
        translator_to_use,
        target_lang_to_use,
        chapter_context=None,
    ):
        # Pass the image and the information of all regions to the translation service for full context
        try:
            results = await self.translation_service.translate_text_batch(
                texts,
                translator=translator_to_use,
                target_lang=target_lang_to_use,
                image=image,
                regions=regions,
                chapter_context=chapter_context,
            )
            pending_updates: list[tuple[int, str]] = []
            for i, result in enumerate(results):
                if result and result.translated_text:
                    pending_updates.append((region_ids[i], result.translated_text))

            if pending_updates:
                self._queue_async_region_updates(
                    pending_updates,
                    field_name="translation",
                    task_kind="translation",
                )
                return

            self._translation_finished.emit("warning", "未生成可应用的翻译结果")
        except Exception as e:
            self.logger.error(f"Translation failed: {e}", exc_info=True)
            self._translation_finished.emit("error", "翻译失败")

    @pyqtSlot(list)
    def set_selection_from_list(self, indices: list):
        """Slot to handle selection changes originating from the RegionListView."""
        self.model.set_selection(indices)

    @pyqtSlot(int, int)
    def move_region_from_list(self, source_index: int, target_index: int):
        """Write a drag-and-drop in the translation list to the model and the undo history."""
        if source_index == target_index:
            return
        region_count = len(self.model.get_regions())
        if not (0 <= source_index < region_count and 0 <= target_index < region_count):
            return
        self.execute_command(
            MoveRegionCommand(
                self.model,
                source_index,
                target_index,
                description=f"Move Region {source_index} to {target_index}",
            )
        )
