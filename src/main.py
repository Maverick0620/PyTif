import os
import copy
import json
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tifffile
from PySide6.QtCore import Qt, QSize, QSettings, QPointF
from PySide6.QtGui import (
    QPixmap,
    QIcon,
    QKeyEvent,
    QAction,
    QPainter,
    QPen,
    QColor,
    QPolygonF,
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
    QToolButton,
    QMenu,
    QFrame,
    QButtonGroup,
    QFormLayout,
)

from utils import (
    natural_key,
    to_8bit_grayscale,
    numpy_to_qimage,
    flatten_to_slices,
    rgb_like_to_gray,
)
from widgets import ImageViewer, DraggablePanel, ROIListWindow
from roi import roi_geometry, roi_mask, serialize_roi_geometry

SUPPORTED_EXTS = (".tif", ".tiff")


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
        self.root_folder: Optional[str] = None  # root of current browsing context
        self.current_folder: Optional[str] = None  # currently displayed folder
        self.entries: List[Tuple[str, str]] = []  # ("up"/"dir"/"tif", path)

        # Image state
        self.loaded: Optional[np.ndarray] = None  # (H,W) or (S,H,W)
        self.total_slices: int = 1
        self.current_slice: int = 0
        self.rois_by_file: Dict[str, List[Dict[str, Any]]] = {}
        self.selected_roi_by_file: Dict[str, int] = {}
        self._updating_rois_from_file = False
        self._updating_roi_list_ui = False

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
        self.btn_file.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_file.clicked.connect(self.open_file_dialog)
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

        self.btn_roi_rect.clicked.connect(
            lambda: self._select_roi_type(ImageViewer.ROI_RECT)
        )
        self.btn_roi_ellipse.clicked.connect(
            lambda: self._select_roi_type(ImageViewer.ROI_ELLIPSE)
        )
        self.btn_roi_poly.clicked.connect(
            lambda: self._select_roi_type(ImageViewer.ROI_POLYGON)
        )

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
            poly = QPolygonF(
                [QPointF(5, 18), QPointF(8, 6), QPointF(18, 8), QPointF(19, 18)]
            )
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
        self.roi_list_widget.currentRowChanged.connect(
            self._on_roi_list_selection_changed
        )
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
            self.root_folder = path
            self.open_folder(path, select_first_tif=True)
            return

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

    def open_folder(
        self,
        folder: str,
        select_first_tif: bool = False,
        select_path: Optional[str] = None,
    ):
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
            [
                n
                for n in names
                if os.path.isfile(os.path.join(folder, n))
                and n.lower().endswith(SUPPORTED_EXTS)
            ],
            key=natural_key,
        )

        self.entries = []
        if self.root_folder and os.path.abspath(folder) != os.path.abspath(
            self.root_folder
        ):
            self.entries.append(("up", os.path.dirname(folder)))

        for d in subdirs:
            self.entries.append(("dir", os.path.join(folder, d)))
        for f in files:
            self.entries.append(("tif", os.path.join(folder, f)))

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

        if select_path:
            target = os.path.abspath(select_path)
            for i, (typ, p) in enumerate(self.entries):
                if typ == "tif" and os.path.abspath(p) == target:
                    self.list_widget.setCurrentRow(i)
                    return
            select_first_tif = True

        if select_first_tif:
            for i, (typ, _) in enumerate(self.entries):
                if typ == "tif":
                    self.list_widget.setCurrentRow(i)
                    return
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
            self.slice_info.setText(
                f"{name} — slice {self.current_slice + 1}/{self.total_slices}"
            )
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
        self.slice_info.setText(
            f"{os.path.basename(self._current_tif_name())} — slice {self.current_slice + 1}/{self.total_slices}"
        )

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
            self.status.setText(
                "ROI mode: choose rectangle, ellipse, or polygon from the floating panel. Press Esc to cancel current drawing."
            )
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
        self.status.setText(
            f"ROI mode: {labels.get(roi_type, '')}. Press Esc to cancel current drawing."
        )

    def _current_file_path(self) -> Optional[str]:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.entries) and self.entries[row][0] == "tif":
            return os.path.abspath(self.entries[row][1])
        return None

    def _on_viewer_rois_changed(
        self, rois: List[Dict[str, Any]], selected_idx: Optional[int]
    ):
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
            if not (isinstance(rid, int) and rid > 0):
                roi["_id"] = next_id
                rid = next_id
                next_id += 1
            name = str(roi.get("name", "")).strip()
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

    def _save_current_file_rois(self):
        path = self._current_file_path()
        if not path:
            self.status.setText("No active TIFF file for ROI save.")
            return
        rois = self.rois_by_file.get(path, [])
        data_rois = []
        for roi in rois:
            s = serialize_roi_geometry(roi)
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
        self.status.setText(
            f"Saved {len(data_rois)} ROI(s) to {os.path.basename(save_path)}"
        )

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

        rois = []
        for r in raw:
            if not isinstance(r, dict):
                continue
            typ = r.get("type")
            if typ == "polygon":
                pts = r.get("points", [])
                if isinstance(pts, list) and len(pts) >= 3:
                    rois.append(
                        {"type": typ, "points": [(float(x), float(y)) for x, y in pts]}
                    )
            elif typ in ("rect", "ellipse"):
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
        self.status.setText(
            f"Loaded {len(rois)} ROI(s) from {os.path.basename(load_path)}"
        )

    def _current_slice_image(self) -> Optional[np.ndarray]:
        if self.loaded is None:
            return None
        if self.loaded.ndim == 2:
            return self.loaded
        return self.loaded[self.current_slice]

    def _update_roi_stats(self):
        path = self._current_file_path()
        rois = self.rois_by_file.get(path, []) if path else []
        self.lbl_roi_count.setText(str(len(rois)))
        state = self.viewer.selected_roi()
        if state is None:
            for lbl in [
                self.lbl_roi_type,
                self.lbl_roi_coords,
                self.lbl_roi_area,
                self.lbl_roi_perimeter,
                self.lbl_roi_pixels,
                self.lbl_roi_mean,
                self.lbl_roi_minmax,
                self.lbl_roi_std,
            ]:
                lbl.setText("—")
            return

        typ_map = {
            "polygon": "Polygon",
            "rect": "Rectangle",
            "ellipse": "Ellipse",
        }
        self.lbl_roi_type.setText(typ_map.get(state.get("type", ""), "Unknown"))
        typ = state.get("type")
        if typ in ("rect", "ellipse"):
            x = float(state.get("x", 0.0))
            y = float(state.get("y", 0.0))
            w = float(state.get("w", 0.0))
            h = float(state.get("h", 0.0))
            self.lbl_roi_coords.setText(f"x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
        elif typ == "polygon":
            pts = state.get("points", [])
            self.lbl_roi_coords.setText(f"{len(pts)} vertices")
        else:
            self.lbl_roi_coords.setText("—")

        area, perimeter = roi_geometry(state)
        self.lbl_roi_area.setText(f"{area:.2f}")
        self.lbl_roi_perimeter.setText(f"{perimeter:.2f}")

        img = self._current_slice_image()
        if img is None:
            for lbl in [
                self.lbl_roi_pixels,
                self.lbl_roi_mean,
                self.lbl_roi_minmax,
                self.lbl_roi_std,
            ]:
                lbl.setText("—")
            return

        mask = roi_mask(state, img.shape)
        vals = img[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            self.lbl_roi_pixels.setText("0")
            for lbl in [self.lbl_roi_mean, self.lbl_roi_minmax, self.lbl_roi_std]:
                lbl.setText("—")
            return

        self.lbl_roi_pixels.setText(str(int(vals.size)))
        self.lbl_roi_mean.setText(f"{float(np.mean(vals)):.4g}")
        self.lbl_roi_minmax.setText(
            f"{float(np.min(vals)):.4g} / {float(np.max(vals)):.4g}"
        )
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

        if key in (Qt.Key_Up, Qt.Key_Down):
            delta = 1 if key == Qt.Key_Down else -1
            self._move_to_prev_next_tif(delta)
            event.accept()
            return

        if key in (Qt.Key_Left, Qt.Key_Right):
            if self.loaded is not None and self.total_slices > 1:
                delta = 1 if key == Qt.Key_Right else -1
                new_slice = max(
                    0, min(self.total_slices - 1, self.current_slice + delta)
                )
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
        if hasattr(self, "roi_panel") and self.roi_panel.isVisible():
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
