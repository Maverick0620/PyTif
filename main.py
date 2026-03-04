import copy
import json
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tifffile

from PySide6.QtCore import Qt, QSize, QEvent, QSettings, QPointF, QLineF, QRectF
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QIcon,
    QPolygonF,
    QKeyEvent,
    QColor,
    QWheelEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QBrush,
    QPainterPath,
    QAction,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsPathItem,
    QGraphicsLineItem,
    QGraphicsEllipseItem,
    QToolButton,
    QMenu,
    QFrame,
    QButtonGroup,
    QFormLayout,
)

SUPPORTED_EXTS = (".tif", ".tiff")


# -------------------------
# Utilities
# -------------------------
def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def to_8bit_grayscale(img2d: np.ndarray) -> np.ndarray:
    x = img2d.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.uint8)

    vmin = np.percentile(x[finite], 1)
    vmax = np.percentile(x[finite], 99)
    if vmax <= vmin:
        vmax = vmin + 1

    y = (x - vmin) / (vmax - vmin)
    y = np.clip(y, 0, 1)
    return (y * 255).astype(np.uint8)


def numpy_to_qimage(img: np.ndarray) -> QImage:
    if img.ndim == 2:
        u8 = to_8bit_grayscale(img)
        h, w = u8.shape
        return QImage(u8.data, w, h, w, QImage.Format_Grayscale8).copy()
    raise ValueError(f"Unsupported image shape: {img.shape}")


def flatten_to_slices(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Return:
      flat: (H,W) if single slice OR (S,H,W) if multi-slice
      slices: number of slices
    """
    if arr.ndim == 2:
        return arr, 1

    h, w = arr.shape[-2:]
    s = int(np.prod(arr.shape[:-2]))
    return arr.reshape(s, h, w), s


def rgb_like_to_gray(arr: np.ndarray) -> np.ndarray:
    """
    Convert common RGB/RGBA TIFF layouts to grayscale.
    Supported examples:
      (H, W, 3/4), (S, H, W, 3/4), (3/4, H, W), (S, 3/4, H, W)
    """
    x = arr

    # Planar channel-first single image: (C,H,W) -> (H,W,C)
    if x.ndim == 3 and x.shape[0] in (3, 4) and x.shape[-1] not in (3, 4):
        x = np.moveaxis(x, 0, -1)

    # Planar channel-first stack: (...,C,H,W) -> (...,H,W,C)
    if x.ndim >= 4 and x.shape[-3] in (3, 4) and x.shape[-1] not in (3, 4):
        x = np.moveaxis(x, -3, -1)

    # Interleaved channel-last RGB/RGBA
    if x.ndim >= 3 and x.shape[-1] in (3, 4):
        rgb = x[..., :3].astype(np.float32, copy=False)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return gray

    return x


# -------------------------
# Image Viewer (Zoom/Pan)
# -------------------------
class DraggablePanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_handle: Optional[QWidget] = None
        self._drag_offset = QPointF()
        self.user_moved = False

    def set_drag_handle(self, widget: QWidget):
        if self._drag_handle is not None:
            self._drag_handle.removeEventFilter(self)
        self._drag_handle = widget
        self._drag_handle.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self._drag_handle:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._drag_offset = event.globalPosition() - self.frameGeometry().topLeft()
                event.accept()
                return True
            if event.type() == QEvent.MouseMove and event.buttons() & Qt.LeftButton:
                parent = self.parentWidget()
                if parent is None:
                    return False
                new_top_left = event.globalPosition() - self._drag_offset
                p = parent.mapFromGlobal(new_top_left.toPoint())
                max_x = max(0, parent.width() - self.width())
                max_y = max(0, parent.height() - self.height())
                self.move(max(0, min(p.x(), max_x)), max(0, min(p.y(), max_y)))
                self.user_moved = True
                event.accept()
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                event.accept()
                return True
        return super().eventFilter(obj, event)


class ROIListWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("ROI List")
        self.resize(300, 420)
        self.on_closed: Optional[Callable[[], None]] = None
        self._closed_by_user = True

        self.setStyleSheet(
            "QWidget { background: rgba(28,28,28,235); color: #ddd; }"
            "QListWidget { background: rgba(20,20,20,220); border: 1px solid #555; }"
            "QPushButton { background: #3a3a3a; border: 1px solid #666; padding: 4px 8px; }"
            "QPushButton:hover { background: #4a4a4a; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(QLabel("ROI List"))

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget, 1)

        row = QHBoxLayout()
        self.btn_save = QPushButton("Save")
        self.btn_load = QPushButton("Load")
        self.btn_close = QPushButton("Close")
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_close)
        layout.addLayout(row)

        self.btn_close.clicked.connect(self.close)

    def hide_programmatically(self):
        self._closed_by_user = False
        self.hide()
        self._closed_by_user = True

    def closeEvent(self, event):
        super().closeEvent(event)
        if self._closed_by_user and self.on_closed:
            self.on_closed()


class ImageViewer(QGraphicsView):
    ROI_NONE = "none"
    ROI_POLYGON = "polygon"
    ROI_RECT = "rect"
    ROI_ELLIPSE = "ellipse"

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)

        self.setRenderHints(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self._has_image = False
        self._min_zoom = 0.05
        self._max_zoom = 40.0
        self._pan_ema = QPointF(0.0, 0.0)
        self._pan_residual = QPointF(0.0, 0.0)

        # ROI state
        self._roi_mode = False
        self._roi_type = self.ROI_NONE
        self._rois: List[Dict[str, Any]] = []
        self._selected_idx: Optional[int] = None
        self._drawing_points: List[QPointF] = []
        self._drawing_hover: Optional[QPointF] = None
        self._shape_start: Optional[QPointF] = None
        self._shape_end: Optional[QPointF] = None
        self._shape_drawing = False

        self._snap_threshold_px = 14.0
        self._vertex_radius = 4.0
        self._vertex_radius_hover = 6.0
        self._edge_hit_threshold_px = 8.0

        self._roi_paths: List[QGraphicsPathItem] = []
        self._roi_preview = QGraphicsLineItem()
        self._roi_preview.setZValue(11)
        self._roi_preview.setPen(QPen(QColor(0, 220, 255), 1.2, Qt.DashLine))
        self._scene.addItem(self._roi_preview)
        self._roi_preview.hide()

        self._drawing_path = QGraphicsPathItem()
        self._drawing_path.setZValue(12)
        self._drawing_path.setPen(QPen(QColor(0, 220, 255), 1.2, Qt.DashLine))
        self._drawing_path.setBrush(QBrush(Qt.NoBrush))
        self._scene.addItem(self._drawing_path)

        self._shape_preview_path = QGraphicsPathItem()
        self._shape_preview_path.setZValue(12)
        self._shape_preview_path.setPen(QPen(QColor(0, 220, 255), 1.4, Qt.DashLine))
        self._shape_preview_path.setBrush(QBrush(QColor(120, 220, 255, 25)))
        self._scene.addItem(self._shape_preview_path)

        self._roi_vertex_items: List[QGraphicsEllipseItem] = []
        self._hover_vertex_idx: Optional[int] = None
        self._drag_vertex_idx: Optional[int] = None
        self._drag_rect_handle: Optional[str] = None
        self._drag_move_roi = False
        self._drag_last_scene: Optional[QPointF] = None
        self._newly_inserted_vertex = False

        self.on_rois_changed: Optional[Callable[[List[Dict[str, Any]], Optional[int]], None]] = None
        self._suppress_notify = False

    def set_image(self, pixmap: QPixmap, fit: bool = True):
        self._pix_item.setPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        self._has_image = not pixmap.isNull()
        if fit:
            self.fit_in_view()

    def fit_in_view(self):
        if not self._has_image:
            return
        self.resetTransform()
        self.fitInView(self._pix_item, Qt.KeepAspectRatio)
        self._clamp_zoom_to_limits()

    def _current_zoom(self) -> float:
        return float(self.transform().m11())

    def _clamp_zoom_to_limits(self):
        cur = self._current_zoom()
        if cur <= 0:
            return
        if cur < self._min_zoom:
            self.scale(self._min_zoom / cur, self._min_zoom / cur)
        elif cur > self._max_zoom:
            self.scale(self._max_zoom / cur, self._max_zoom / cur)

    def _apply_zoom_factor(self, factor: float):
        if not self._has_image:
            return
        if factor <= 0:
            return
        cur = self._current_zoom()
        target = max(self._min_zoom, min(self._max_zoom, cur * factor))
        actual = target / max(cur, 1e-12)
        self.scale(actual, actual)

    def _reset_pan_smoothing(self):
        self._pan_ema = QPointF(0.0, 0.0)
        self._pan_residual = QPointF(0.0, 0.0)

    def _apply_smooth_pan(self, pd):
        dx = float(pd.x())
        dy = float(pd.y())
        speed = math.hypot(dx, dy)

        # Nonlinear gain: precise at low speed, accelerated at high speed.
        if speed < 1.0:
            gain = 0.55
        elif speed < 6.0:
            gain = 0.75
        elif speed < 18.0:
            gain = 1.00
        else:
            gain = 1.25

        # EMA smoothing. At higher speed we reduce smoothing to keep responsiveness.
        alpha = 0.42 if speed < 8.0 else 0.62
        target_x = dx * gain
        target_y = dy * gain
        fx = alpha * target_x + (1.0 - alpha) * float(self._pan_ema.x())
        fy = alpha * target_y + (1.0 - alpha) * float(self._pan_ema.y())
        self._pan_ema = QPointF(fx, fy)

        # Accumulate sub-pixel remainder to avoid jitter/stair-stepping.
        tx = fx + float(self._pan_residual.x())
        ty = fy + float(self._pan_residual.y())
        sx = int(tx)
        sy = int(ty)
        self._pan_residual = QPointF(tx - sx, ty - sy)

        if sx != 0:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - sx)
        if sy != 0:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - sy)

    def zoom_in(self):
        if self._has_image:
            self._apply_zoom_factor(1.15)

    def zoom_out(self):
        if self._has_image:
            self._apply_zoom_factor(1 / 1.15)

    def set_roi_mode(self, enabled: bool):
        self._roi_mode = enabled
        self._shape_drawing = False
        self._drawing_points = []
        self._drawing_hover = None
        self._hover_vertex_idx = None
        self._drag_vertex_idx = None
        self._drag_rect_handle = None
        self._drag_move_roi = False
        self._drag_last_scene = None
        if enabled:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.unsetCursor()
        self._update_roi_graphics()

    def set_roi_type(self, roi_type: str):
        if roi_type not in (self.ROI_NONE, self.ROI_POLYGON, self.ROI_RECT, self.ROI_ELLIPSE):
            return
        self._roi_type = roi_type
        self.cancel_current_roi()
        self._update_roi_graphics()

    def roi_type(self) -> str:
        return self._roi_type

    def clear_roi(self, notify: bool = True):
        self._rois = []
        self._selected_idx = None
        self._drawing_points = []
        self._drawing_hover = None
        self._hover_vertex_idx = None
        self._shape_start = None
        self._shape_end = None
        self._shape_drawing = False
        self._drag_vertex_idx = None
        self._drag_rect_handle = None
        self._drag_move_roi = False
        self._drag_last_scene = None
        self._update_roi_graphics()
        if notify:
            self._notify_rois_changed()

    def cancel_current_roi(self):
        if self._drawing_points:
            self._drawing_points = []
            self._drawing_hover = None
            self._hover_vertex_idx = None
            self._update_roi_graphics()
        if self._roi_type in (self.ROI_RECT, self.ROI_ELLIPSE) and self._shape_drawing:
            self._shape_drawing = False
            self._shape_start = None
            self._shape_end = None
            self._update_roi_graphics()
        self._drag_vertex_idx = None
        self._drag_rect_handle = None
        self._drag_move_roi = False
        self._drag_last_scene = None

    def clear_all_rois(self):
        self.clear_roi(notify=True)

    def get_rois(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._rois)

    def selected_roi(self) -> Optional[Dict[str, Any]]:
        if self._selected_idx is None:
            return None
        if 0 <= self._selected_idx < len(self._rois):
            return copy.deepcopy(self._rois[self._selected_idx])
        return None

    def selected_roi_index(self) -> Optional[int]:
        return self._selected_idx

    def set_rois(self, rois: List[Dict[str, Any]], selected_idx: Optional[int] = None):
        self._suppress_notify = True
        self.clear_roi(notify=False)
        self._rois = copy.deepcopy(rois)
        if selected_idx is not None and 0 <= selected_idx < len(self._rois):
            self._selected_idx = selected_idx
        elif self._rois:
            self._selected_idx = len(self._rois) - 1
        else:
            self._selected_idx = None
        self._update_roi_graphics()
        self._suppress_notify = False
        self._notify_rois_changed()

    def _notify_rois_changed(self):
        if self._suppress_notify:
            return
        if self.on_rois_changed:
            self.on_rois_changed(self.get_rois(), self._selected_idx)

    def _clamp_to_image(self, p: QPointF) -> QPointF:
        if self._pix_item.pixmap().isNull():
            return p
        rect = self._pix_item.boundingRect()
        x = min(max(p.x(), rect.left()), rect.right())
        y = min(max(p.y(), rect.top()), rect.bottom())
        return QPointF(x, y)

    def _find_snap_idx(self, points: List[QPointF], mouse_pos_view) -> Optional[int]:
        if not points:
            return None
        best_idx = None
        best_dist = None
        for i, p in enumerate(points):
            pv = self.mapFromScene(p)
            dx = pv.x() - mouse_pos_view.x()
            dy = pv.y() - mouse_pos_view.y()
            d = (dx * dx + dy * dy) ** 0.5
            if d <= self._snap_threshold_px and (best_dist is None or d < best_dist):
                best_dist = d
                best_idx = i
        return best_idx

    def _roi_path(self, roi: Dict[str, Any]) -> QPainterPath:
        path = QPainterPath()
        typ = roi.get("type")
        if typ == self.ROI_POLYGON:
            pts = roi.get("points", [])
            if len(pts) >= 3:
                first = QPointF(float(pts[0][0]), float(pts[0][1]))
                path.moveTo(first)
                for x, y in pts[1:]:
                    path.lineTo(QPointF(float(x), float(y)))
                path.closeSubpath()
        elif typ in (self.ROI_RECT, self.ROI_ELLIPSE):
            x = float(roi.get("x", 0.0))
            y = float(roi.get("y", 0.0))
            w = float(roi.get("w", 0.0))
            h = float(roi.get("h", 0.0))
            rect = QRectF(x, y, w, h).normalized()
            if typ == self.ROI_RECT:
                path.addRect(rect)
            else:
                path.addEllipse(rect)
        return path

    def _current_selected_points(self) -> List[QPointF]:
        if self._selected_idx is None or self._selected_idx >= len(self._rois):
            return []
        roi = self._rois[self._selected_idx]
        typ = roi.get("type")
        if typ == self.ROI_POLYGON:
            return [QPointF(float(x), float(y)) for x, y in roi.get("points", [])]
        if typ in (self.ROI_RECT, self.ROI_ELLIPSE):
            rect = QRectF(float(roi.get("x", 0.0)), float(roi.get("y", 0.0)), float(roi.get("w", 0.0)), float(roi.get("h", 0.0))).normalized()
            return [
                rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft(),
                QPointF(rect.center().x(), rect.top()),
                QPointF(rect.right(), rect.center().y()),
                QPointF(rect.center().x(), rect.bottom()),
                QPointF(rect.left(), rect.center().y()),
            ]
        return []

    def _shape_rect(self) -> Optional[QRectF]:
        if self._shape_start is None or self._shape_end is None:
            return None
        return QRectF(self._shape_start, self._shape_end).normalized()

    def _set_rect(self, rect: QRectF):
        self._shape_start = rect.topLeft()
        self._shape_end = rect.bottomRight()

    def _point_segment_distance_scene(self, p: QPointF, a: QPointF, b: QPointF) -> float:
        apx = p.x() - a.x()
        apy = p.y() - a.y()
        abx = b.x() - a.x()
        aby = b.y() - a.y()
        ab2 = abx * abx + aby * aby
        if ab2 <= 1e-9:
            return QLineF(p, a).length()
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
        proj = QPointF(a.x() + t * abx, a.y() + t * aby)
        return QLineF(p, proj).length()

    def _find_polygon_edge_idx(self, mouse_pos_view) -> Optional[int]:
        if self._selected_idx is None or self._selected_idx >= len(self._rois):
            return None
        roi = self._rois[self._selected_idx]
        if roi.get("type") != self.ROI_POLYGON:
            return None
        pts = [QPointF(float(x), float(y)) for x, y in roi.get("points", [])]
        if len(pts) < 3:
            return None
        scene_p = self.mapToScene(mouse_pos_view)
        best_idx = None
        best_d = None
        n = len(pts)
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            d_scene = self._point_segment_distance_scene(scene_p, a, b)
            d_view = d_scene * self.transform().m11()
            if d_view <= self._edge_hit_threshold_px and (best_d is None or d_view < best_d):
                best_d = d_view
                best_idx = i
        return best_idx

    def _update_vertex_items(self):
        points: List[QPointF] = []
        style_type = None
        if self._roi_mode and self._roi_type == self.ROI_POLYGON and self._drawing_points:
            points = self._drawing_points
            style_type = self.ROI_POLYGON
        else:
            points = self._current_selected_points()
            if self._selected_idx is not None and self._selected_idx < len(self._rois):
                style_type = self._rois[self._selected_idx].get("type")

        while len(self._roi_vertex_items) < len(points):
            item = QGraphicsEllipseItem()
            item.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
            item.setZValue(12)
            self._scene.addItem(item)
            self._roi_vertex_items.append(item)
        while len(self._roi_vertex_items) > len(points):
            item = self._roi_vertex_items.pop()
            self._scene.removeItem(item)

        for i, p in enumerate(points):
            item = self._roi_vertex_items[i]
            is_hover = (style_type == self.ROI_POLYGON and self._hover_vertex_idx == i)
            radius = self._vertex_radius_hover if is_hover else self._vertex_radius
            if style_type == self.ROI_POLYGON:
                # Keep start-point visually distinct for easier closure.
                color = QColor(0, 220, 255) if is_hover else QColor(255, 120, 0 if i == 0 else 255)
            else:
                color = QColor(90, 210, 255)
            item.setRect(-radius, -radius, 2 * radius, 2 * radius)
            item.setPos(p)
            item.setPen(QPen(Qt.black, 1))
            item.setBrush(QBrush(color))
            item.setVisible(True)

    def _update_roi_graphics(self):
        while len(self._roi_paths) < len(self._rois):
            item = QGraphicsPathItem()
            item.setZValue(10)
            self._scene.addItem(item)
            self._roi_paths.append(item)
        while len(self._roi_paths) > len(self._rois):
            item = self._roi_paths.pop()
            self._scene.removeItem(item)

        for i, roi in enumerate(self._rois):
            path = self._roi_path(roi)
            item = self._roi_paths[i]
            selected = (self._selected_idx == i)
            if selected:
                item.setPen(QPen(QColor(255, 210, 60), 2.2))
                item.setBrush(QBrush(QColor(255, 190, 0, 60)))
            else:
                item.setPen(QPen(QColor(140, 220, 255), 1.4))
                item.setBrush(QBrush(QColor(80, 170, 230, 30)))
            item.setPath(path)

        if self._roi_mode and self._roi_type == self.ROI_POLYGON and self._drawing_points and self._drawing_hover is not None:
            self._roi_preview.setLine(QLineF(self._drawing_points[-1], self._drawing_hover))
            self._roi_preview.show()
        else:
            self._roi_preview.hide()

        dpath = QPainterPath()
        if self._drawing_points:
            dpath.moveTo(self._drawing_points[0])
            for p in self._drawing_points[1:]:
                dpath.lineTo(p)
        self._drawing_path.setPath(dpath)

        spath = QPainterPath()
        if self._roi_mode and self._shape_drawing and self._shape_start is not None and self._shape_end is not None:
            rect = QRectF(self._shape_start, self._shape_end).normalized()
            if self._roi_type == self.ROI_RECT:
                spath.addRect(rect)
            elif self._roi_type == self.ROI_ELLIPSE:
                spath.addEllipse(rect)
        self._shape_preview_path.setPath(spath)

        self._update_vertex_items()

    def mousePressEvent(self, event: QMouseEvent):
        if self._roi_mode and self._has_image:
            mouse = event.position().toPoint()
            scene_p = self._clamp_to_image(self.mapToScene(mouse))

            if event.button() == Qt.RightButton:
                hit = self._hit_test_roi(scene_p)
                if hit is not None:
                    if self._selected_idx == hit:
                        del self._rois[hit]
                        if self._selected_idx is not None:
                            self._selected_idx = min(hit, len(self._rois) - 1) if self._rois else None
                        self._update_roi_graphics()
                        self._notify_rois_changed()
                    else:
                        self._selected_idx = hit
                        self._update_roi_graphics()
                        self._notify_rois_changed()
                    event.accept()
                    return

            if event.button() == Qt.LeftButton:
                if self._selected_idx is not None and self._selected_idx < len(self._rois):
                    roi = self._rois[self._selected_idx]
                    if roi.get("type") == self.ROI_POLYGON:
                        pts = [QPointF(float(x), float(y)) for x, y in roi.get("points", [])]
                        idx = self._find_snap_idx(pts, mouse)
                        if idx is not None:
                            self._drag_vertex_idx = idx
                            self._drag_last_scene = scene_p
                            event.accept()
                            return
                        edge_idx = self._find_polygon_edge_idx(mouse)
                        if edge_idx is not None:
                            pts.insert(edge_idx + 1, scene_p)
                            roi["points"] = [(p.x(), p.y()) for p in pts]
                            self._drag_vertex_idx = edge_idx + 1
                            self._newly_inserted_vertex = True
                            self._drag_last_scene = scene_p
                            self._update_roi_graphics()
                            event.accept()
                            return
                    if roi.get("type") in (self.ROI_RECT, self.ROI_ELLIPSE):
                        handle = self._hit_rect_handle(roi, mouse)
                        if handle is not None:
                            self._drag_rect_handle = handle
                            self._shape_drawing = True
                            self._drag_last_scene = scene_p
                            event.accept()
                            return
                    if self._roi_path(roi).contains(scene_p):
                        self._drag_move_roi = True
                        self._drag_last_scene = scene_p
                        event.accept()
                        return

                hit = self._hit_test_roi(scene_p)
                if hit is not None:
                    self._selected_idx = hit
                    self._update_roi_graphics()
                    self._notify_rois_changed()
                    event.accept()
                    return

                if self._roi_type == self.ROI_POLYGON:
                    snap_idx = self._find_snap_idx(self._drawing_points, mouse)
                    if snap_idx is not None:
                        scene_p = self._drawing_points[snap_idx]
                    if not self._drawing_points:
                        self._drawing_points.append(scene_p)
                    elif snap_idx == 0 and len(self._drawing_points) >= 3:
                        roi = {"type": self.ROI_POLYGON, "points": [(p.x(), p.y()) for p in self._drawing_points]}
                        self._rois.append(roi)
                        self._selected_idx = len(self._rois) - 1
                        self._drawing_points = []
                        self._drawing_hover = None
                        self._hover_vertex_idx = None
                        self._notify_rois_changed()
                    else:
                        if QLineF(self._drawing_points[-1], scene_p).length() > 1e-6:
                            self._drawing_points.append(scene_p)
                    self._update_roi_graphics()
                    event.accept()
                    return

                if self._roi_type in (self.ROI_RECT, self.ROI_ELLIPSE):
                    self._shape_start = scene_p
                    self._shape_end = scene_p
                    self._shape_drawing = True
                    self._drag_rect_handle = None
                    self._update_roi_graphics()
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._roi_mode and self._has_image:
            mouse = event.position().toPoint()
            scene_p = self._clamp_to_image(self.mapToScene(mouse))
            if self._drag_move_roi and self._selected_idx is not None and self._drag_last_scene is not None:
                dx = scene_p.x() - self._drag_last_scene.x()
                dy = scene_p.y() - self._drag_last_scene.y()
                self._move_roi_by(self._selected_idx, dx, dy)
                self._drag_last_scene = scene_p
                self._update_roi_graphics()
                return
            if self._drag_vertex_idx is not None and self._selected_idx is not None:
                roi = self._rois[self._selected_idx]
                pts = [QPointF(float(x), float(y)) for x, y in roi.get("points", [])]
                if 0 <= self._drag_vertex_idx < len(pts):
                    pts[self._drag_vertex_idx] = scene_p
                    roi["points"] = [(p.x(), p.y()) for p in pts]
                    self._update_roi_graphics()
                return
            if self._drag_rect_handle is not None and self._selected_idx is not None:
                roi = self._rois[self._selected_idx]
                self._resize_roi_handle(roi, self._drag_rect_handle, scene_p)
                self._update_roi_graphics()
                return
            if self._shape_drawing and self._shape_start is not None:
                self._shape_end = scene_p
                self._update_roi_graphics()
                return
            if self._drawing_points:
                snap_idx = self._find_snap_idx(self._drawing_points, mouse)
                self._hover_vertex_idx = snap_idx
                self._drawing_hover = self._drawing_points[snap_idx] if snap_idx is not None else scene_p
                self._update_roi_graphics()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._roi_mode and event.button() == Qt.LeftButton:
            if self._drag_move_roi:
                self._drag_move_roi = False
                self._drag_last_scene = None
                self._notify_rois_changed()
                event.accept()
                return
            if self._drag_vertex_idx is not None:
                self._drag_vertex_idx = None
                self._newly_inserted_vertex = False
                self._notify_rois_changed()
                event.accept()
                return
            if self._drag_rect_handle is not None:
                self._drag_rect_handle = None
                self._shape_drawing = False
                self._notify_rois_changed()
                event.accept()
                return
            if self._shape_drawing and self._shape_start is not None and self._shape_end is not None:
                rect = QRectF(self._shape_start, self._shape_end).normalized()
                self._shape_drawing = False
                if rect.width() >= 1.0 and rect.height() >= 1.0:
                    roi = {"type": self._roi_type, "x": rect.x(), "y": rect.y(), "w": rect.width(), "h": rect.height()}
                    self._rois.append(roi)
                    self._selected_idx = len(self._rois) - 1
                    self._notify_rois_changed()
                self._shape_start = None
                self._shape_end = None
                self._update_roi_graphics()
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if self._roi_mode and self._drawing_points:
            self._hover_vertex_idx = None
            self._drawing_hover = None
            self._update_roi_graphics()
        super().leaveEvent(event)

    def _hit_test_roi(self, scene_p: QPointF) -> Optional[int]:
        for i in range(len(self._rois) - 1, -1, -1):
            if self._roi_path(self._rois[i]).contains(scene_p):
                return i
        return None

    def _rect_from_roi(self, roi: Dict[str, Any]) -> QRectF:
        return QRectF(float(roi.get("x", 0.0)), float(roi.get("y", 0.0)), float(roi.get("w", 0.0)), float(roi.get("h", 0.0))).normalized()

    def _hit_rect_handle(self, roi: Dict[str, Any], mouse_pos_view) -> Optional[str]:
        rect = self._rect_from_roi(roi)
        if rect.width() < 1.0 or rect.height() < 1.0:
            return None
        handles = {
            "tl": rect.topLeft(),
            "tr": rect.topRight(),
            "br": rect.bottomRight(),
            "bl": rect.bottomLeft(),
            "t": QPointF(rect.center().x(), rect.top()),
            "r": QPointF(rect.right(), rect.center().y()),
            "b": QPointF(rect.center().x(), rect.bottom()),
            "l": QPointF(rect.left(), rect.center().y()),
        }
        for name, hp in handles.items():
            vp = self.mapFromScene(hp)
            d = ((vp.x() - mouse_pos_view.x()) ** 2 + (vp.y() - mouse_pos_view.y()) ** 2) ** 0.5
            if d <= self._snap_threshold_px:
                return name
        return None

    def _resize_roi_handle(self, roi: Dict[str, Any], handle: str, scene_p: QPointF):
        rect = self._rect_from_roi(roi)
        left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
        if handle in ("tl", "bl", "l"):
            left = scene_p.x()
        if handle in ("tr", "br", "r"):
            right = scene_p.x()
        if handle in ("tl", "tr", "t"):
            top = scene_p.y()
        if handle in ("bl", "br", "b"):
            bottom = scene_p.y()
        nr = QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()
        roi["x"], roi["y"], roi["w"], roi["h"] = nr.x(), nr.y(), nr.width(), nr.height()

    def _move_roi_by(self, idx: int, dx: float, dy: float):
        if not (0 <= idx < len(self._rois)):
            return
        roi = self._rois[idx]
        if roi.get("type") == self.ROI_POLYGON:
            roi["points"] = [(float(x) + dx, float(y) + dy) for x, y in roi.get("points", [])]
        else:
            roi["x"] = float(roi.get("x", 0.0)) + dx
            roi["y"] = float(roi.get("y", 0.0)) + dy

    def nudge_selected_roi(self, dx: float, dy: float) -> bool:
        if self._selected_idx is None:
            return False
        self._move_roi_by(self._selected_idx, dx, dy)
        self._update_roi_graphics()
        self._notify_rois_changed()
        return True

    def wheelEvent(self, event: QWheelEvent):
        # Ctrl / Cmd + wheel/trackpad scroll => zoom.
        if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            self._reset_pan_smoothing()
            delta = event.angleDelta().y()
            if delta == 0:
                delta = event.pixelDelta().y()
            if delta > 0:
                # Smooth wheel scaling.
                self._apply_zoom_factor(1.0015 ** float(delta))
            elif delta < 0:
                self._apply_zoom_factor(1.0015 ** float(delta))
            event.accept()
            return

        # Ignore synthesized pinch-as-wheel events (no modifier), so gesture
        # zoom is effectively disabled and won't cause jitter.
        source = event.source()
        src_system = getattr(Qt, "MouseEventSynthesizedBySystem", None)
        if src_system is None and hasattr(Qt, "MouseEventSource"):
            src_system = Qt.MouseEventSource.MouseEventSynthesizedBySystem
        if (
            src_system is not None
            and source == src_system
            and event.modifiers() == Qt.NoModifier
            and event.pixelDelta().isNull()
            and event.angleDelta().y() != 0
        ):
            self._reset_pan_smoothing()
            event.accept()
            return

        # Trackpad two-finger pan in both browse and ROI modes.
        pd = event.pixelDelta()
        if not pd.isNull():
            self._apply_smooth_pan(pd)
            event.accept()
            return
        self._reset_pan_smoothing()
        super().wheelEvent(event)

    def event(self, e):
        return super().event(e)


# -------------------------
# Main Window
# -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyTIF Viewer")
        self.resize(1400, 850)
        self.setAcceptDrops(True)

        self.settings = QSettings("PyTIF", "Viewer")

        # Navigation state
        self.root_folder: Optional[str] = None     # root of current browsing context
        self.current_folder: Optional[str] = None  # currently displayed folder
        self.entries: List[Tuple[str, str]] = []   # ("up"/"dir"/"tif", path)

        # Image state
        self.loaded: Optional[np.ndarray] = None   # (H,W) or (S,H,W)
        self.total_slices: int = 1
        self.current_slice: int = 0
        self.rois_by_file: Dict[str, List[Dict[str, Any]]] = {}
        self.selected_roi_by_file: Dict[str, int] = {}
        self._updating_rois_from_file = False
        self._updating_roi_list_ui = False

        # thumbs not included in this lean version (you already had it earlier)
        self._build_ui()
        self._build_menu()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        layout.addLayout(top)

        # File dropdown button
        self.btn_file = QToolButton()
        self.btn_file.setText("Open")
        self.btn_file.setPopupMode(QToolButton.InstantPopup)
        self.file_dropdown = QMenu(self.btn_file)
        self.file_dropdown.addAction("Open File", self.open_file_dialog)
        self.file_dropdown.addAction("Open Folder", self.open_folder_dialog)
        self.file_dropdown.addAction("Add Files", self.add_files_dialog)
        self.btn_file.setMenu(self.file_dropdown)
        top.addWidget(self.btn_file)

        self.btn_sidebar = QPushButton("Hide Sidebar")
        self.btn_sidebar.clicked.connect(self.toggle_sidebar)
        top.addWidget(self.btn_sidebar)

        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_in = QPushButton("+")
        self.btn_fit = QPushButton("Fit")
        self.btn_roi = QPushButton("ROI")
        self.btn_roi.setCheckable(True)
        top.addWidget(self.btn_zoom_out)
        top.addWidget(self.btn_zoom_in)
        top.addWidget(self.btn_fit)
        top.addWidget(self.btn_roi)

        self.status = QLabel("")
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.status, 1)

        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        self.btn_fit.clicked.connect(self._fit)
        self.btn_roi.toggled.connect(self._toggle_roi_mode)
        self._build_roi_panel()

        self.splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.splitter, 1)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_entry_selected)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.splitter.addWidget(self.list_widget)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(1, 1)

        self.viewer = ImageViewer()
        self.viewer.on_rois_changed = self._on_viewer_rois_changed
        right_layout.addWidget(self.viewer, 1)

        # Slice controls (only for multi-slice)
        self.slice_controls = QWidget()
        sl = QHBoxLayout(self.slice_controls)
        sl.setContentsMargins(0, 0, 0, 0)

        self.slice_info = QLabel("")
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_spin = QSpinBox()

        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_spin.setMinimum(1)
        self.slice_spin.setMaximum(1)

        sl.addWidget(self.slice_info, 1)
        sl.addWidget(self.slice_slider, 4)
        sl.addWidget(QLabel("Slice"))
        sl.addWidget(self.slice_spin)

        right_layout.addWidget(self.slice_controls)
        self.slice_controls.hide()
        self._build_roi_list_window()
        self._build_roi_stats_panel(right_layout)

        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        self.slice_spin.valueChanged.connect(self.on_spin_changed)

    def _build_menu(self):
        # macOS standard menu
        file_menu = self.menuBar().addMenu("File")

        act_open_file = QAction("Open File…", self)
        act_open_file.setShortcut("Meta+O")  # ⌘O
        act_open_file.triggered.connect(self.open_file_dialog)
        file_menu.addAction(act_open_file)

        act_open_folder = QAction("Open Folder…", self)
        act_open_folder.triggered.connect(self.open_folder_dialog)
        file_menu.addAction(act_open_folder)

        act_add_files = QAction("Add Files…", self)
        act_add_files.triggered.connect(self.add_files_dialog)
        file_menu.addAction(act_add_files)

        file_menu.addSeparator()

        act_close = QAction("Close Window", self)
        act_close.setShortcut("Meta+W")  # ⌘W
        act_close.triggered.connect(self.close)
        file_menu.addAction(act_close)

    def _build_roi_panel(self):
        self.roi_panel = DraggablePanel(self)
        self.roi_panel.setFrameShape(QFrame.StyledPanel)
        self.roi_panel.setObjectName("roiPanel")
        self.roi_panel.setStyleSheet(
            "#roiPanel { background: rgba(40,40,40,220); border: 1px solid #666; border-radius: 8px; }"
            "#roiPanel QLabel { color: #ddd; }"
            "#roiPanel QToolButton { background: transparent; border: 1px solid #888; border-radius: 6px; padding: 2px; }"
            "#roiPanel QToolButton:checked { background: #2d7bd8; border-color: #64a7ff; }"
        )
        panel_layout = QVBoxLayout(self.roi_panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(6)

        title = QLabel("ROI Tools")
        title.setCursor(Qt.SizeAllCursor)
        self.roi_panel.set_drag_handle(title)
        panel_layout.addWidget(title)

        self.lbl_keep_mode = QLabel("Keep ROI in Current File: On")
        panel_layout.addWidget(self.lbl_keep_mode)

        row = QHBoxLayout()

        self.btn_roi_rect = QToolButton()
        self.btn_roi_rect.setCheckable(True)
        self.btn_roi_rect.setIcon(self._make_roi_icon(ImageViewer.ROI_RECT))
        self.btn_roi_rect.setIconSize(QSize(20, 20))
        self.btn_roi_rect.setFixedSize(34, 34)
        self.btn_roi_rect.setToolTip("Rectangle ROI")

        self.btn_roi_ellipse = QToolButton()
        self.btn_roi_ellipse.setCheckable(True)
        self.btn_roi_ellipse.setIcon(self._make_roi_icon(ImageViewer.ROI_ELLIPSE))
        self.btn_roi_ellipse.setIconSize(QSize(20, 20))
        self.btn_roi_ellipse.setFixedSize(34, 34)
        self.btn_roi_ellipse.setToolTip("Ellipse ROI")

        self.btn_roi_poly = QToolButton()
        self.btn_roi_poly.setCheckable(True)
        self.btn_roi_poly.setIcon(self._make_roi_icon(ImageViewer.ROI_POLYGON))
        self.btn_roi_poly.setIconSize(QSize(20, 20))
        self.btn_roi_poly.setFixedSize(34, 34)
        self.btn_roi_poly.setToolTip("Polygon ROI")

        self.roi_type_group = QButtonGroup(self)
        self.roi_type_group.setExclusive(True)
        self.roi_type_group.addButton(self.btn_roi_rect)
        self.roi_type_group.addButton(self.btn_roi_ellipse)
        self.roi_type_group.addButton(self.btn_roi_poly)

        row.addWidget(self.btn_roi_rect)
        row.addWidget(self.btn_roi_ellipse)
        row.addWidget(self.btn_roi_poly)
        panel_layout.addLayout(row)

        self.btn_roi_rect.clicked.connect(lambda: self._select_roi_type(ImageViewer.ROI_RECT))
        self.btn_roi_ellipse.clicked.connect(lambda: self._select_roi_type(ImageViewer.ROI_ELLIPSE))
        self.btn_roi_poly.clicked.connect(lambda: self._select_roi_type(ImageViewer.ROI_POLYGON))

        self.btn_clear_rois = QPushButton("Clear All ROIs")
        self.btn_clear_rois.clicked.connect(self._clear_current_file_rois)
        panel_layout.addWidget(self.btn_clear_rois)

        self.roi_panel.hide()

    def _make_roi_icon(self, roi_type: str) -> QIcon:
        pm = QPixmap(24, 24)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(235, 235, 235), 2))
        if roi_type == ImageViewer.ROI_RECT:
            p.drawRect(4, 5, 16, 14)
        elif roi_type == ImageViewer.ROI_ELLIPSE:
            p.drawEllipse(4, 5, 16, 14)
        else:
            poly = QPolygonF([QPointF(5, 18), QPointF(8, 6), QPointF(18, 8), QPointF(19, 18)])
            p.drawPolygon(poly)
        p.end()
        return QIcon(pm)

    def _build_roi_stats_panel(self, parent_layout: QVBoxLayout):
        self.roi_stats_panel = QFrame()
        self.roi_stats_panel.setObjectName("roiStatsPanel")
        self.roi_stats_panel.setStyleSheet(
            "#roiStatsPanel { border: 1px solid #666; border-radius: 6px; background: rgba(30,30,30,140); }"
            "#roiStatsPanel QLabel { color: #ddd; }"
        )
        stats_layout = QFormLayout(self.roi_stats_panel)
        stats_layout.setContentsMargins(8, 6, 8, 6)
        stats_layout.setLabelAlignment(Qt.AlignRight)
        stats_layout.setHorizontalSpacing(12)
        stats_layout.setVerticalSpacing(4)

        self.lbl_roi_type = QLabel("—")
        self.lbl_roi_count = QLabel("0")
        self.lbl_roi_coords = QLabel("—")
        self.lbl_roi_area = QLabel("—")
        self.lbl_roi_perimeter = QLabel("—")
        self.lbl_roi_pixels = QLabel("—")
        self.lbl_roi_mean = QLabel("—")
        self.lbl_roi_minmax = QLabel("—")
        self.lbl_roi_std = QLabel("—")

        stats_layout.addRow("ROI Type", self.lbl_roi_type)
        stats_layout.addRow("ROIs in File", self.lbl_roi_count)
        stats_layout.addRow("Coordinates", self.lbl_roi_coords)
        stats_layout.addRow("Area (px²)", self.lbl_roi_area)
        stats_layout.addRow("Perimeter (px)", self.lbl_roi_perimeter)
        stats_layout.addRow("Pixel Count", self.lbl_roi_pixels)
        stats_layout.addRow("Mean", self.lbl_roi_mean)
        stats_layout.addRow("Min / Max", self.lbl_roi_minmax)
        stats_layout.addRow("Std Dev", self.lbl_roi_std)

        parent_layout.addWidget(self.roi_stats_panel)

    def _build_roi_list_window(self):
        self.roi_window = ROIListWindow(self)
        self.roi_list_widget = self.roi_window.list_widget
        self.roi_list_widget.currentRowChanged.connect(self._on_roi_list_selection_changed)
        self.roi_list_widget.itemChanged.connect(self._on_roi_list_item_changed)
        self.roi_window.btn_save.clicked.connect(self._save_current_file_rois)
        self.roi_window.btn_load.clicked.connect(self._load_rois_into_current_file)
        self.roi_window.on_closed = self._on_roi_window_closed

    # ---------------- Open ----------------
    def open_file_dialog(self):
        start = self.current_folder or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TIFF File",
            start,
            "TIFF files (*.tif *.tiff)",
        )
        if path:
            self.open_path(path)

    def add_files_dialog(self):
        start = self.current_folder or os.path.expanduser("~")
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add TIFF Files",
            start,
            "TIFF files (*.tif *.tiff)",
        )
        if paths:
            self.add_files(paths)

    def add_files(self, paths: List[str]):
        existing = {os.path.abspath(p) for typ, p in self.entries if typ == "tif"}
        added_indices: List[int] = []
        for p in paths:
            ap = os.path.abspath(p)
            if not os.path.isfile(ap):
                continue
            if not ap.lower().endswith(SUPPORTED_EXTS):
                continue
            if ap in existing:
                continue
            self.entries.append(("tif", ap))
            item = QListWidgetItem(os.path.basename(ap))
            item.setToolTip(ap)
            self.list_widget.addItem(item)
            added_indices.append(self.list_widget.count() - 1)
            existing.add(ap)

        if not added_indices:
            self.status.setText("No new TIFF files were added.")
            return

        if self.loaded is None:
            self.list_widget.setCurrentRow(added_indices[0])
        self.status.setText(f"Added {len(added_indices)} file(s).")

    def open_folder_dialog(self):
        start = self.current_folder or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(
            self,
            "Open Folder",
            start,
        )
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        path = os.path.abspath(path)
        if os.path.isdir(path):
            # Folder chosen => root is this folder
            self.root_folder = path
            self.open_folder(path, select_first_tif=True)
            return

        # File chosen
        if os.path.isfile(path):
            self.open_single_file(path)
            return

        self.status.setText(f"Path does not exist: {path}")

    # ---------------- Folder browsing ----------------
    def open_single_file(self, path: str):
        path = os.path.abspath(path)
        folder = os.path.dirname(path)
        self.root_folder = folder
        self.current_folder = folder
        self.entries = [("tif", path)]

        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        item = QListWidgetItem(os.path.basename(path))
        item.setToolTip(path)
        self.list_widget.addItem(item)
        self.list_widget.setCurrentRow(0)
        self.list_widget.blockSignals(False)

        self.load_tiff(path)

    def open_folder(self, folder: str, select_first_tif: bool = False, select_path: Optional[str] = None):
        folder = os.path.abspath(folder)
        self.current_folder = folder

        try:
            names = os.listdir(folder)
        except Exception as e:
            self.status.setText(f"Failed to list folder: {folder} ({e})")
            return

        subdirs = sorted(
            [n for n in names if os.path.isdir(os.path.join(folder, n))],
            key=natural_key,
        )
        files = sorted(
            [n for n in names if os.path.isfile(os.path.join(folder, n)) and n.lower().endswith(SUPPORTED_EXTS)],
            key=natural_key,
        )

        self.entries = []
        # Add ".." (go up) if not at root
        if self.root_folder and os.path.abspath(folder) != os.path.abspath(self.root_folder):
            self.entries.append(("up", os.path.dirname(folder)))

        for d in subdirs:
            self.entries.append(("dir", os.path.join(folder, d)))
        for f in files:
            self.entries.append(("tif", os.path.join(folder, f)))

        # Refresh list
        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        for typ, p in self.entries:
            if typ == "up":
                text = "⬅︎  .. (Back)"
            elif typ == "dir":
                text = f"📁  {os.path.basename(p)}"
            else:
                text = os.path.basename(p)

            item = QListWidgetItem(text)
            item.setToolTip(p)
            self.list_widget.addItem(item)

        self.list_widget.blockSignals(False)

        self.status.setText(f"Folder: {folder}")

        # Select item
        if select_path:
            target = os.path.abspath(select_path)
            for i, (typ, p) in enumerate(self.entries):
                if typ == "tif" and os.path.abspath(p) == target:
                    self.list_widget.setCurrentRow(i)
                    return
            # If not found, fallback
            select_first_tif = True

        if select_first_tif:
            for i, (typ, _) in enumerate(self.entries):
                if typ == "tif":
                    self.list_widget.setCurrentRow(i)
                    return
            # No tif in folder: keep selection at 0 if exists
            if self.entries:
                self.list_widget.setCurrentRow(0)

    def on_item_double_clicked(self, item: QListWidgetItem):
        row = self.list_widget.row(item)
        if row < 0 or row >= len(self.entries):
            return

        typ, path = self.entries[row]
        if typ in ("dir", "up"):
            self.open_folder(path, select_first_tif=True)
        elif typ == "tif":
            self.load_tiff(path)

    def on_entry_selected(self, row: int):
        if row < 0 or row >= len(self.entries):
            return
        typ, path = self.entries[row]
        if typ == "tif":
            self.load_tiff(path)

    # ---------------- TIFF load/render ----------------
    def load_tiff(self, path: str):
        try:
            arr = tifffile.imread(path)
            arr = rgb_like_to_gray(arr)
            flat, slices = flatten_to_slices(arr)
        except Exception as e:
            self.status.setText(f"Failed to load {os.path.basename(path)}: {e}")
            return

        self.loaded = flat
        self.total_slices = slices
        self.current_slice = 0

        # Slice UI only if slices > 1
        if slices > 1:
            self.slice_controls.show()

            self.slice_slider.blockSignals(True)
            self.slice_spin.blockSignals(True)

            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(slices - 1)
            self.slice_slider.setValue(0)

            self.slice_spin.setMinimum(1)
            self.slice_spin.setMaximum(slices)
            self.slice_spin.setValue(1)

            self.slice_slider.blockSignals(False)
            self.slice_spin.blockSignals(False)
        else:
            self.slice_controls.hide()

        self._render(fit=True)
        self._apply_rois_for_current_file()
        self._update_roi_stats()
        self._update_slice_info(path)

    def _update_slice_info(self, path: str):
        name = os.path.basename(path)
        if self.total_slices > 1:
            self.slice_info.setText(f"{name} — slice {self.current_slice + 1}/{self.total_slices}")
        else:
            self.slice_info.setText(f"{name} — 2D")

        if self.current_folder:
            self.status.setText(f"Folder: {self.current_folder}  |  {name}")

    def _render(self, fit: bool = False):
        if self.loaded is None:
            return

        if self.loaded.ndim == 2:
            img = self.loaded
        else:
            img = self.loaded[self.current_slice]

        qimg = numpy_to_qimage(img)
        pix = QPixmap.fromImage(qimg)

        # Keep zoom when switching slices; fit only on new file / explicit fit
        self.viewer.set_image(pix, fit=fit)

    # ---------------- Slice ----------------
    def on_slice_changed(self, v: int):
        if self.loaded is None or self.total_slices <= 1:
            return

        self.current_slice = int(v)

        self.slice_spin.blockSignals(True)
        self.slice_spin.setValue(self.current_slice + 1)
        self.slice_spin.blockSignals(False)

        self._render(fit=False)
        self._update_roi_stats()
        self.slice_info.setText(f"{os.path.basename(self._current_tif_name())} — slice {self.current_slice + 1}/{self.total_slices}")

    def on_spin_changed(self, v: int):
        if self.loaded is None or self.total_slices <= 1:
            return
        self.slice_slider.setValue(int(v) - 1)

    def _current_tif_name(self) -> str:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.entries) and self.entries[row][0] == "tif":
            return self.entries[row][1]
        return ""

    # ---------------- Sidebar ----------------
    def toggle_sidebar(self):
        if self.list_widget.isVisible():
            self.list_widget.hide()
            self.btn_sidebar.setText("Show Sidebar")
        else:
            self.list_widget.show()
            self.btn_sidebar.setText("Hide Sidebar")

    # ---------------- Zoom helpers ----------------
    def _zoom_in(self):
        self.viewer.zoom_in()

    def _zoom_out(self):
        self.viewer.zoom_out()

    def _fit(self):
        self.viewer.fit_in_view()

    def _toggle_roi_mode(self, enabled: bool):
        self.viewer.set_roi_mode(enabled)
        if enabled:
            self._sync_roi_type_buttons()
            self._show_roi_panel()
            self._show_roi_window()
            self.btn_roi.setText("ROI On")
            self.status.setText("ROI mode: choose rectangle, ellipse, or polygon from the floating panel. Press Esc to cancel current drawing.")
        else:
            self._hide_roi_panel()
            self._hide_roi_window()
            self.btn_roi_rect.setChecked(False)
            self.btn_roi_ellipse.setChecked(False)
            self.btn_roi_poly.setChecked(False)
            self.btn_roi.setText("ROI")

    def _show_roi_panel(self):
        self.roi_panel.adjustSize()
        if not self.roi_panel.user_moved or not self.roi_panel.isVisible():
            anchor = self.btn_roi.mapTo(self, self.btn_roi.rect().bottomLeft())
            x = anchor.x()
            y = anchor.y() + 6
            self.roi_panel.move(x, y)
        self._clamp_roi_panel_pos()
        self.roi_panel.show()
        self.roi_panel.raise_()

    def _hide_roi_panel(self):
        self.roi_panel.hide()

    def _show_roi_window(self):
        if not self.roi_window.isVisible():
            p = self.roi_window.pos()
            if p.x() == 0 and p.y() == 0:
                g = self.geometry()
                self.roi_window.move(g.right() + 12, g.top() + 60)
        self.roi_window.show()
        self.roi_window.raise_()
        self.roi_window.activateWindow()

    def _hide_roi_window(self):
        self.roi_window.hide_programmatically()

    def _on_roi_window_closed(self):
        if self.btn_roi.isChecked():
            self.btn_roi.setChecked(False)

    def _clamp_roi_panel_pos(self):
        max_x = max(0, self.width() - self.roi_panel.width() - 8)
        max_y = max(0, self.height() - self.roi_panel.height() - 8)
        x = max(0, min(self.roi_panel.x(), max_x))
        y = max(0, min(self.roi_panel.y(), max_y))
        self.roi_panel.move(x, y)

    def _sync_roi_type_buttons(self):
        t = self.viewer.roi_type()
        self.btn_roi_rect.setChecked(t == ImageViewer.ROI_RECT)
        self.btn_roi_ellipse.setChecked(t == ImageViewer.ROI_ELLIPSE)
        self.btn_roi_poly.setChecked(t == ImageViewer.ROI_POLYGON)

    def _select_roi_type(self, roi_type: str):
        if not self.btn_roi.isChecked():
            self.btn_roi.setChecked(True)
        self.viewer.set_roi_type(roi_type)
        labels = {
            ImageViewer.ROI_RECT: "Rectangle ROI",
            ImageViewer.ROI_ELLIPSE: "Ellipse ROI",
            ImageViewer.ROI_POLYGON: "Polygon ROI",
        }
        self.status.setText(f"ROI mode: {labels.get(roi_type, '')}. Press Esc to cancel current drawing.")

    def _current_file_path(self) -> Optional[str]:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.entries) and self.entries[row][0] == "tif":
            return os.path.abspath(self.entries[row][1])
        return None

    def _on_viewer_rois_changed(self, rois: List[Dict[str, Any]], selected_idx: Optional[int]):
        if self._updating_rois_from_file:
            return
        path = self._current_file_path()
        if not path:
            return
        self._ensure_roi_ids_and_names(path, rois)
        self.rois_by_file[path] = copy.deepcopy(rois)
        if selected_idx is None:
            self.selected_roi_by_file.pop(path, None)
        else:
            self.selected_roi_by_file[path] = int(selected_idx)
        self._refresh_roi_list()
        self._update_roi_stats()

    def _apply_rois_for_current_file(self):
        path = self._current_file_path()
        rois = copy.deepcopy(self.rois_by_file.get(path, [])) if path else []
        if path:
            self._ensure_roi_ids_and_names(path, rois)
            self.rois_by_file[path] = copy.deepcopy(rois)
        sel = self.selected_roi_by_file.get(path) if path else None
        self._updating_rois_from_file = True
        self.viewer.set_rois(rois, sel)
        self._updating_rois_from_file = False
        self._refresh_roi_list()
        self._sync_roi_type_buttons()

    def _clear_current_file_rois(self):
        path = self._current_file_path()
        if not path:
            return
        self.rois_by_file[path] = []
        self.selected_roi_by_file.pop(path, None)
        self._apply_rois_for_current_file()
        self._update_roi_stats()

    def _ensure_roi_ids_and_names(self, path: str, rois: List[Dict[str, Any]]):
        next_id = 1
        for roi in rois:
            rid = roi.get("_id")
            if isinstance(rid, int) and rid > 0:
                next_id = max(next_id, rid + 1)

        for roi in rois:
            rid = roi.get("_id")
            # Assign ID only once for newly-created ROI objects.
            if not (isinstance(rid, int) and rid > 0):
                roi["_id"] = next_id
                rid = next_id
                next_id += 1
            name = str(roi.get("name", "")).strip()
            # Default display name is bound to stable internal ID.
            if not name:
                roi["name"] = str(rid)

    def _refresh_roi_list(self):
        path = self._current_file_path()
        rois = self.rois_by_file.get(path, []) if path else []
        sel = self.selected_roi_by_file.get(path) if path else None

        self._updating_roi_list_ui = True
        self.roi_list_widget.clear()
        for i, roi in enumerate(rois):
            rid = roi.get("_id")
            default_name = str(rid) if isinstance(rid, int) and rid > 0 else str(i + 1)
            name = str(roi.get("name", default_name))
            typ = str(roi.get("type", ""))
            rid_text = str(rid) if isinstance(rid, int) and rid > 0 else "?"
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setData(Qt.UserRole, i)
            item.setToolTip(f"ID {rid_text} · {name} ({typ})")
            self.roi_list_widget.addItem(item)
        if sel is not None and 0 <= sel < self.roi_list_widget.count():
            self.roi_list_widget.setCurrentRow(sel)
        self._updating_roi_list_ui = False

    def _on_roi_list_selection_changed(self, row: int):
        if self._updating_roi_list_ui:
            return
        path = self._current_file_path()
        if not path:
            return
        rois = copy.deepcopy(self.rois_by_file.get(path, []))
        sel = row if 0 <= row < len(rois) else None
        if sel is None:
            self.selected_roi_by_file.pop(path, None)
        else:
            self.selected_roi_by_file[path] = sel
        self._updating_rois_from_file = True
        self.viewer.set_rois(rois, sel)
        self._updating_rois_from_file = False
        self._update_roi_stats()

    def _on_roi_list_item_changed(self, item: QListWidgetItem):
        if self._updating_roi_list_ui:
            return
        path = self._current_file_path()
        if not path:
            return
        idx = self.roi_list_widget.row(item)
        rois = self.rois_by_file.get(path, [])
        if not (0 <= idx < len(rois)):
            return
        rid = rois[idx].get("_id", idx + 1)
        name = item.text().strip() or str(rid)
        rois[idx]["name"] = name
        item.setText(name)
        item.setToolTip(f"ID {rid} · {name} ({rois[idx].get('type', '')})")
        sel = self.selected_roi_by_file.get(path)
        self._updating_rois_from_file = True
        self.viewer.set_rois(copy.deepcopy(rois), sel)
        self._updating_rois_from_file = False

    def _serialize_roi_geometry(self, roi: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        typ = roi.get("type")
        if typ == ImageViewer.ROI_POLYGON:
            pts = roi.get("points", [])
            return {
                "type": typ,
                "points": [[float(x), float(y)] for x, y in pts],
            }
        if typ in (ImageViewer.ROI_RECT, ImageViewer.ROI_ELLIPSE):
            return {
                "type": typ,
                "x": float(roi.get("x", 0.0)),
                "y": float(roi.get("y", 0.0)),
                "w": float(roi.get("w", 0.0)),
                "h": float(roi.get("h", 0.0)),
            }
        return None

    def _save_current_file_rois(self):
        path = self._current_file_path()
        if not path:
            self.status.setText("No active TIFF file for ROI save.")
            return
        rois = self.rois_by_file.get(path, [])
        data_rois = []
        for roi in rois:
            s = self._serialize_roi_geometry(roi)
            if s is not None:
                data_rois.append(s)

        default_name = os.path.splitext(os.path.basename(path))[0] + ".roi.json"
        default_path = os.path.join(os.path.dirname(path), default_name)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save ROI File",
            default_path,
            "ROI JSON (*.json)",
        )
        if not save_path:
            return
        payload = {
            "version": 1,
            "image_path": os.path.abspath(path),
            "image_name": os.path.basename(path),
            "rois": data_rois,
        }
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            self.status.setText(f"Failed to save ROI file: {e}")
            return
        self.status.setText(f"Saved {len(data_rois)} ROI(s) to {os.path.basename(save_path)}")

    def _load_rois_into_current_file(self):
        path = self._current_file_path()
        if not path:
            self.status.setText("No active TIFF file for ROI load.")
            return
        load_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load ROI File",
            os.path.dirname(path),
            "ROI JSON (*.json)",
        )
        if not load_path:
            return
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            self.status.setText(f"Failed to load ROI file: {e}")
            return

        raw = payload.get("rois", [])
        if not isinstance(raw, list):
            self.status.setText("Invalid ROI file: 'rois' must be a list.")
            return

        rois: List[Dict[str, Any]] = []
        for r in raw:
            if not isinstance(r, dict):
                continue
            typ = r.get("type")
            if typ == ImageViewer.ROI_POLYGON:
                pts = r.get("points", [])
                if isinstance(pts, list) and len(pts) >= 3:
                    rois.append({"type": typ, "points": [(float(x), float(y)) for x, y in pts]})
            elif typ in (ImageViewer.ROI_RECT, ImageViewer.ROI_ELLIPSE):
                rois.append(
                    {
                        "type": typ,
                        "x": float(r.get("x", 0.0)),
                        "y": float(r.get("y", 0.0)),
                        "w": float(r.get("w", 0.0)),
                        "h": float(r.get("h", 0.0)),
                    }
                )

        self.rois_by_file[path] = rois
        self.selected_roi_by_file[path] = 0 if rois else None
        if not rois:
            self.selected_roi_by_file.pop(path, None)
        self._apply_rois_for_current_file()
        self._update_roi_stats()
        self.status.setText(f"Loaded {len(rois)} ROI(s) from {os.path.basename(load_path)}")

    def _current_slice_image(self) -> Optional[np.ndarray]:
        if self.loaded is None:
            return None
        if self.loaded.ndim == 2:
            return self.loaded
        return self.loaded[self.current_slice]

    def _roi_geometry(self, state: Dict[str, Any]) -> Tuple[float, float]:
        typ = state.get("type")
        if typ == ImageViewer.ROI_POLYGON:
            pts = np.array(state.get("points", []), dtype=np.float64)
            if len(pts) < 3:
                return 0.0, 0.0
            x = pts[:, 0]
            y = pts[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            perimeter = float(np.sum(np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))))
            return float(area), perimeter
        if typ == ImageViewer.ROI_RECT:
            w = float(state.get("w", 0.0))
            h = float(state.get("h", 0.0))
            return w * h, 2.0 * (w + h)
        if typ == ImageViewer.ROI_ELLIPSE:
            a = float(state.get("w", 0.0)) / 2.0
            b = float(state.get("h", 0.0)) / 2.0
            area = np.pi * a * b
            perimeter = float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))) if a > 0 and b > 0 else 0.0
            return float(area), perimeter
        return 0.0, 0.0

    def _roi_mask(self, state: Dict[str, Any], shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        typ = state.get("type")
        mask = np.zeros((h, w), dtype=bool)
        if typ == ImageViewer.ROI_RECT:
            x0 = float(state.get("x", 0.0))
            y0 = float(state.get("y", 0.0))
            rw = float(state.get("w", 0.0))
            rh = float(state.get("h", 0.0))
            xmin = max(0, int(np.floor(min(x0, x0 + rw))))
            xmax = min(w, int(np.ceil(max(x0, x0 + rw))))
            ymin = max(0, int(np.floor(min(y0, y0 + rh))))
            ymax = min(h, int(np.ceil(max(y0, y0 + rh))))
            if xmin >= xmax or ymin >= ymax:
                return mask
            yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
            x = xx + 0.5
            y = yy + 0.5
            mask[ymin:ymax, xmin:xmax] = (x >= x0) & (x <= x0 + rw) & (y >= y0) & (y <= y0 + rh)
            return mask
        if typ == ImageViewer.ROI_ELLIPSE:
            x0 = float(state.get("x", 0.0))
            y0 = float(state.get("y", 0.0))
            rw = float(state.get("w", 0.0))
            rh = float(state.get("h", 0.0))
            xmin = max(0, int(np.floor(min(x0, x0 + rw))))
            xmax = min(w, int(np.ceil(max(x0, x0 + rw))))
            ymin = max(0, int(np.floor(min(y0, y0 + rh))))
            ymax = min(h, int(np.ceil(max(y0, y0 + rh))))
            if xmin >= xmax or ymin >= ymax:
                return mask
            yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
            x = xx + 0.5
            y = yy + 0.5
            cx = x0 + rw / 2.0
            cy = y0 + rh / 2.0
            a = max(rw / 2.0, 1e-9)
            b = max(rh / 2.0, 1e-9)
            mask[ymin:ymax, xmin:xmax] = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0
            return mask
        if typ == ImageViewer.ROI_POLYGON:
            pts = np.array(state.get("points", []), dtype=np.float64)
            if len(pts) < 3:
                return mask
            xmin = max(0, int(np.floor(np.min(pts[:, 0]))))
            xmax = min(w, int(np.ceil(np.max(pts[:, 0]))))
            ymin = max(0, int(np.floor(np.min(pts[:, 1]))))
            ymax = min(h, int(np.ceil(np.max(pts[:, 1]))))
            if xmin >= xmax or ymin >= ymax:
                return mask
            yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
            x = xx + 0.5
            y = yy + 0.5
            px = pts[:, 0]
            py = pts[:, 1]
            inside = np.zeros((ymax - ymin, xmax - xmin), dtype=bool)
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                xi, yi = px[i], py[i]
                xj, yj = px[j], py[j]
                intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)
                inside ^= intersect
            mask[ymin:ymax, xmin:xmax] = inside
            return mask
        return mask

    def _update_roi_stats(self):
        path = self._current_file_path()
        rois = self.rois_by_file.get(path, []) if path else []
        self.lbl_roi_count.setText(str(len(rois)))
        state = self.viewer.selected_roi()
        if state is None:
            self.lbl_roi_type.setText("—")
            self.lbl_roi_coords.setText("—")
            self.lbl_roi_area.setText("—")
            self.lbl_roi_perimeter.setText("—")
            self.lbl_roi_pixels.setText("—")
            self.lbl_roi_mean.setText("—")
            self.lbl_roi_minmax.setText("—")
            self.lbl_roi_std.setText("—")
            return

        typ_map = {
            ImageViewer.ROI_POLYGON: "Polygon",
            ImageViewer.ROI_RECT: "Rectangle",
            ImageViewer.ROI_ELLIPSE: "Ellipse",
        }
        self.lbl_roi_type.setText(typ_map.get(state.get("type"), "Unknown"))
        typ = state.get("type")
        if typ in (ImageViewer.ROI_RECT, ImageViewer.ROI_ELLIPSE):
            x = float(state.get("x", 0.0))
            y = float(state.get("y", 0.0))
            w = float(state.get("w", 0.0))
            h = float(state.get("h", 0.0))
            self.lbl_roi_coords.setText(f"x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
        elif typ == ImageViewer.ROI_POLYGON:
            pts = state.get("points", [])
            self.lbl_roi_coords.setText(f"{len(pts)} vertices")
        else:
            self.lbl_roi_coords.setText("—")

        area, perimeter = self._roi_geometry(state)
        self.lbl_roi_area.setText(f"{area:.2f}")
        self.lbl_roi_perimeter.setText(f"{perimeter:.2f}")

        img = self._current_slice_image()
        if img is None:
            self.lbl_roi_pixels.setText("—")
            self.lbl_roi_mean.setText("—")
            self.lbl_roi_minmax.setText("—")
            self.lbl_roi_std.setText("—")
            return

        mask = self._roi_mask(state, img.shape)
        vals = img[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            self.lbl_roi_pixels.setText("0")
            self.lbl_roi_mean.setText("—")
            self.lbl_roi_minmax.setText("—")
            self.lbl_roi_std.setText("—")
            return

        self.lbl_roi_pixels.setText(str(int(vals.size)))
        self.lbl_roi_mean.setText(f"{float(np.mean(vals)):.4g}")
        self.lbl_roi_minmax.setText(f"{float(np.min(vals)):.4g} / {float(np.max(vals)):.4g}")
        self.lbl_roi_std.setText(f"{float(np.std(vals)):.4g}")

    # ---------------- Drag & Drop ----------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if not path:
            return
        self.open_path(path)

    # ---------------- Keyboard ----------------
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()

        # Arrow keys nudge selected ROI by 1 px in ROI mode.
        if self.btn_roi.isChecked():
            if key == Qt.Key_Left and self.viewer.nudge_selected_roi(-1, 0):
                event.accept()
                return
            if key == Qt.Key_Right and self.viewer.nudge_selected_roi(1, 0):
                event.accept()
                return
            if key == Qt.Key_Up and self.viewer.nudge_selected_roi(0, -1):
                event.accept()
                return
            if key == Qt.Key_Down and self.viewer.nudge_selected_roi(0, 1):
                event.accept()
                return

        # Up/Down => file only (skip dirs/up)
        if key in (Qt.Key_Up, Qt.Key_Down):
            delta = 1 if key == Qt.Key_Down else -1
            self._move_to_prev_next_tif(delta)
            event.accept()
            return

        # Left/Right => slices only
        if key in (Qt.Key_Left, Qt.Key_Right):
            if self.loaded is not None and self.total_slices > 1:
                delta = 1 if key == Qt.Key_Right else -1
                new_slice = max(0, min(self.total_slices - 1, self.current_slice + delta))
                if new_slice != self.current_slice:
                    self.slice_slider.setValue(new_slice)
            event.accept()
            return

        if key == Qt.Key_Escape and self.btn_roi.isChecked():
            self.viewer.cancel_current_roi()
            event.accept()
            return

        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.roi_panel.isVisible():
            self._show_roi_panel()

    def _move_to_prev_next_tif(self, delta: int):
        if not self.entries:
            return
        cur = self.list_widget.currentRow()
        if cur < 0:
            return

        idx = cur
        while True:
            idx += delta
            if idx < 0 or idx >= len(self.entries):
                break
            if self.entries[idx][0] == "tif":
                self.list_widget.setCurrentRow(idx)
                break


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
