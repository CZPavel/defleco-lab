from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class RoiItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, rect: QtCore.QRectF, roi_type: str, name: str) -> None:
        super().__init__(rect)
        self.roi_type, self.name = roi_type, name
        self.roi_enabled = True
        color = QtGui.QColor("#4dd0e1" if roi_type == "analysis" else "#ffb74d")
        self.setPen(QtGui.QPen(color, 2, QtCore.Qt.PenStyle.DashLine))
        self.setBrush(QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue(), 28)))
        self.setFlags(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setToolTip(
            f"{roi_type.title()} ROI: {name}. Select and use +/- to resize, Delete to remove."
        )


class ImageViewer(QtWidgets.QGraphicsView):
    pixelHovered = QtCore.Signal(int, int, str)
    roiChanged = QtCore.Signal()
    viewTransformChanged = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pixmap = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap)
        self._image: np.ndarray | None = None
        self._fit = True
        self._roi_mode: str | None = None
        self._origin: QtCore.QPointF | None = None
        self._rubber: QtWidgets.QGraphicsRectItem | None = None
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QtGui.QColor("#11151a"))

    def set_array(self, image: np.ndarray, heatmap: bool = False) -> None:
        a = np.asarray(image)
        if a.ndim == 2:
            if a.dtype != np.uint8:
                finite = np.nan_to_num(a.astype(np.float32))
                lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
                a = np.clip((finite - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)
            if heatmap:
                import cv2

                a = cv2.cvtColor(cv2.applyColorMap(a, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            else:
                a = np.ascontiguousarray(a)
                qimage = QtGui.QImage(
                    a.data,
                    a.shape[1],
                    a.shape[0],
                    a.strides[0],
                    QtGui.QImage.Format.Format_Grayscale8,
                ).copy()
                self._set_qimage(qimage, image)
                return
        if a.ndim == 3 and a.shape[2] == 3:
            a = np.ascontiguousarray(a)
            qimage = QtGui.QImage(
                a.data, a.shape[1], a.shape[0], a.strides[0], QtGui.QImage.Format.Format_RGB888
            ).copy()
            self._set_qimage(qimage, image)

    def _set_qimage(self, qimage: QtGui.QImage, source: np.ndarray) -> None:
        self._image = np.asarray(source).copy()
        self._pixmap.setPixmap(QtGui.QPixmap.fromImage(qimage))
        self.scene().setSceneRect(self._pixmap.boundingRect())
        if self._fit:
            self.fit_to_window()

    def fit_to_window(self) -> None:
        if not self._pixmap.pixmap().isNull():
            self.fitInView(self._pixmap, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self._fit = True
        self.viewTransformChanged.emit(self.transform())

    def actual_size(self) -> None:
        self.resetTransform()
        self._fit = False
        self.viewTransformChanged.emit(self.transform())

    def apply_view_transform(self, transform: QtGui.QTransform) -> None:
        self.setTransform(transform)
        self._fit = False

    def reset_view(self) -> None:
        self.fit_to_window()

    def begin_roi(self, roi_type: str) -> None:
        self._roi_mode = roi_type
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)

    def rois(self, roi_type: str | None = None) -> list[dict[str, object]]:
        items = []
        for item in self.scene().items():
            if isinstance(item, RoiItem) and (roi_type is None or item.roi_type == roi_type):
                rect = item.mapRectToScene(item.rect())
                items.append(
                    {
                        "name": item.name,
                        "type": item.roi_type,
                        "x": int(rect.x()),
                        "y": int(rect.y()),
                        "width": int(rect.width()),
                        "height": int(rect.height()),
                        "enabled": item.roi_enabled,
                    }
                )
        return items

    def delete_selected(self) -> None:
        for item in self.scene().selectedItems():
            if isinstance(item, RoiItem):
                self.scene().removeItem(item)
        self.roiChanged.emit()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        item = self.itemAt(event.pos())
        if not isinstance(item, RoiItem):
            super().contextMenuEvent(event)
            return
        menu = QtWidgets.QMenu(self)
        rename = menu.addAction("Rename ROI")
        toggle = menu.addAction("Disable ROI" if item.roi_enabled else "Enable ROI")
        delete = menu.addAction("Delete ROI")
        selected = menu.exec(event.globalPos())
        if selected is rename:
            name, ok = QtWidgets.QInputDialog.getText(self, "Rename ROI", "Name", text=item.name)
            if ok and name.strip():
                item.name = name.strip()
                item.setToolTip(f"{item.roi_type.title()} ROI: {item.name}")
        elif selected is toggle:
            item.roi_enabled = not item.roi_enabled
            item.setOpacity(1.0 if item.roi_enabled else 0.35)
        elif selected is delete:
            self.scene().removeItem(item)
        self.roiChanged.emit()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Delete:
            self.delete_selected()
            return
        if event.key() in (QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Minus):
            delta = 8 if event.key() == QtCore.Qt.Key.Key_Plus else -8
            for item in self.scene().selectedItems():
                if isinstance(item, RoiItem):
                    r = item.rect()
                    r.setWidth(max(16, r.width() + delta))
                    r.setHeight(max(16, r.height() + delta))
                    item.setRect(r)
            self.roiChanged.emit()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)
        self._fit = False
        self.viewTransformChanged.emit(self.transform())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._roi_mode and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._origin = self.mapToScene(event.position().toPoint())
            self._rubber = QtWidgets.QGraphicsRectItem(QtCore.QRectF(self._origin, self._origin))
            self._rubber.setPen(QtGui.QPen(QtGui.QColor("white"), 1, QtCore.Qt.PenStyle.DotLine))
            self.scene().addItem(self._rubber)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        point = self.mapToScene(event.position().toPoint())
        if self._rubber is not None and self._origin is not None:
            self._rubber.setRect(QtCore.QRectF(self._origin, point).normalized())
        elif self._image is not None:
            x, y = int(point.x()), int(point.y())
            if 0 <= y < self._image.shape[0] and 0 <= x < self._image.shape[1]:
                self.pixelHovered.emit(x, y, str(self._image[y, x]))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._rubber is not None:
            rect = self._rubber.rect().intersected(self._pixmap.boundingRect())
            self.scene().removeItem(self._rubber)
            self._rubber = None
            if rect.width() >= 16 and rect.height() >= 16:
                count = len(self.rois(self._roi_mode)) + 1
                self.scene().addItem(RoiItem(rect, self._roi_mode or "analysis", f"ROI {count}"))
                self.roiChanged.emit()
            self._origin = None
            self._roi_mode = None
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            return
        super().mouseReleaseEvent(event)
