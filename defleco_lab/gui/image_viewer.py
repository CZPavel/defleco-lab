from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class _RoiHandle(QtWidgets.QGraphicsRectItem):
    SIZE = 10.0

    def __init__(self, corner: str, parent: RoiItem) -> None:
        half = self.SIZE / 2.0
        super().__init__(-half, -half, self.SIZE, self.SIZE, parent)
        self.corner = corner
        self.setBrush(QtGui.QColor("#f2f5f7"))
        self.setPen(QtGui.QPen(QtGui.QColor("#11151a"), 1))
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )
        self.setZValue(10)
        self.setCursor(
            QtCore.Qt.CursorShape.SizeFDiagCursor
            if corner in {"tl", "br"}
            else QtCore.Qt.CursorShape.SizeBDiagCursor
        )

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        event.accept()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        parent = self.parentItem()
        if isinstance(parent, RoiItem):
            parent.resize_from_handle(self.corner, parent.mapFromScene(event.scenePos()))
        event.accept()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        parent = self.parentItem()
        if isinstance(parent, RoiItem):
            parent.notify_changed()
        event.accept()


class RoiItem(QtWidgets.QGraphicsRectItem):
    MIN_SIZE = 16.0

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
        self.setZValue(5)
        self._label = QtWidgets.QGraphicsSimpleTextItem(self)
        self._label.setBrush(color.lighter(130))
        self._label.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )
        self._handles = {
            corner: _RoiHandle(corner, self)
            for corner in ("tl", "tr", "bl", "br")
        }
        self._update_visuals()
        self.setToolTip(
            f"{roi_type.title()} ROI: {name}. Drag the box to move it; "
            "drag a white corner handle to resize; Delete removes it."
        )

    def _update_visuals(self) -> None:
        rect = self.rect()
        label_type = "ANALYSIS" if self.roi_type == "analysis" else "MOTION"
        self._label.setText(f"{label_type}: {self.name}")
        self._label.setPos(rect.topLeft() + QtCore.QPointF(6.0, 6.0))
        for corner, point in {
            "tl": rect.topLeft(),
            "tr": rect.topRight(),
            "bl": rect.bottomLeft(),
            "br": rect.bottomRight(),
        }.items():
            self._handles[corner].setPos(point)

    def resize_from_handle(self, corner: str, point: QtCore.QPointF) -> None:
        rect = QtCore.QRectF(self.rect())
        bounds = (
            self.mapRectFromScene(self.scene().sceneRect())
            if self.scene() is not None
            else QtCore.QRectF()
        )
        x, y = point.x(), point.y()
        if not bounds.isNull():
            x = min(max(x, bounds.left()), bounds.right())
            y = min(max(y, bounds.top()), bounds.bottom())
        if "l" in corner:
            rect.setLeft(min(x, rect.right() - self.MIN_SIZE))
        else:
            rect.setRight(max(x, rect.left() + self.MIN_SIZE))
        if "t" in corner:
            rect.setTop(min(y, rect.bottom() - self.MIN_SIZE))
        else:
            rect.setBottom(max(y, rect.top() + self.MIN_SIZE))
        self.setRect(rect)
        self._update_visuals()

    def notify_changed(self) -> None:
        scene = self.scene()
        if scene is None:
            return
        for view in scene.views():
            if isinstance(view, ImageViewer):
                view.roiChanged.emit()

    def itemChange(self, change, value):
        if (
            change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and self.scene() is not None
        ):
            bounds = self.scene().sceneRect()
            rect = self.rect()
            if not bounds.isNull():
                x = min(max(value.x(), bounds.left() - rect.left()), bounds.right() - rect.right())
                y = min(max(value.y(), bounds.top() - rect.top()), bounds.bottom() - rect.bottom())
                return QtCore.QPointF(x, y)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        self.notify_changed()


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
        self._last_fit_key: tuple[int, int, int, int] | None = None
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QtGui.QColor("#11151a"))
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(160, 120)
        self._set_scrollbars_for_fit(True)

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
        # Keep a reference for pixel inspection; QImage/QPixmap already own the display copy.
        # Avoid another multi-megapixel memcpy on every live frame.
        self._image = np.asarray(source)
        self._pixmap.setPixmap(QtGui.QPixmap.fromImage(qimage))
        self.scene().setSceneRect(self._pixmap.boundingRect())
        self._refit_if_needed()

    def clear_image(self) -> None:
        self._image = None
        self._pixmap.setPixmap(QtGui.QPixmap())
        self.scene().setSceneRect(QtCore.QRectF())
        self._last_fit_key = None
        if self._fit:
            self.resetTransform()

    def _set_scrollbars_for_fit(self, fit: bool) -> None:
        policy = (
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            if fit
            else QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.setHorizontalScrollBarPolicy(policy)
        self.setVerticalScrollBarPolicy(policy)

    def _refit_if_needed(self, force: bool = False) -> None:
        if not self._fit or self._pixmap.pixmap().isNull():
            return
        viewport = self.viewport()
        pixmap = self._pixmap.pixmap()
        key = (viewport.width(), viewport.height(), pixmap.width(), pixmap.height())
        if not force and key == self._last_fit_key:
            return
        if key[0] < 2 or key[1] < 2 or key[2] < 1 or key[3] < 1:
            return

        # Compute the fit explicitly instead of QGraphicsView.fitInView().
        # Qt's fitInView() can toggle automatic scrollbars while resizing, which
        # changes the viewport size and can leave a fitted image clipped.
        self._set_scrollbars_for_fit(True)
        self.resetTransform()
        rect = self._pixmap.boundingRect()
        available_width = max(1.0, float(viewport.width() - 2))
        available_height = max(1.0, float(viewport.height() - 2))
        scale = min(
            available_width / max(rect.width(), 1.0),
            available_height / max(rect.height(), 1.0),
        )
        if np.isfinite(scale) and scale > 0:
            self.scale(scale, scale)
            self.centerOn(rect.center())
        self._last_fit_key = key
        self.viewTransformChanged.emit((QtGui.QTransform(self.transform()), self._fit))

    def refresh_fit(self) -> None:
        """Refit only when this viewer is already in Fit mode."""
        self._refit_if_needed(force=True)

    def fit_to_window(self) -> None:
        self._fit = True
        self._last_fit_key = None
        self._refit_if_needed(force=True)

    def actual_size(self) -> None:
        self._fit = False
        self._last_fit_key = None
        self._set_scrollbars_for_fit(False)
        self.resetTransform()
        if not self._pixmap.pixmap().isNull():
            self.centerOn(self._pixmap.boundingRect().center())
        self.viewTransformChanged.emit((QtGui.QTransform(self.transform()), self._fit))

    def apply_view_transform(self, state: object) -> None:
        # Compare viewers propagate both the transform and whether it represents
        # automatic Fit mode. Keeping only the matrix made one side silently leave
        # Fit mode whenever the other side refitted.
        fit = False
        transform = state
        if isinstance(state, tuple) and len(state) == 2:
            transform, fit = state
        if not isinstance(transform, QtGui.QTransform):
            return
        self._fit = bool(fit)
        self._last_fit_key = None
        self._set_scrollbars_for_fit(self._fit)
        self.setTransform(transform)

    def reset_view(self) -> None:
        self.fit_to_window()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._fit:
            QtCore.QTimer.singleShot(0, self._refit_if_needed)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._fit:
            QtCore.QTimer.singleShot(0, self._refit_if_needed)

    def begin_roi(self, roi_type: str) -> None:
        self._roi_mode = roi_type
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def add_default_roi(self, roi_type: str) -> bool:
        if self._pixmap.pixmap().isNull():
            return False
        bounds = self._pixmap.boundingRect()
        width = min(max(64.0, bounds.width() * 0.35), bounds.width() * 0.8)
        height = min(max(64.0, bounds.height() * 0.35), bounds.height() * 0.8)
        rect = QtCore.QRectF(
            bounds.center().x() - width / 2.0,
            bounds.center().y() - height / 2.0,
            width,
            height,
        )
        for selected in self.scene().selectedItems():
            selected.setSelected(False)
        count = len(self.rois(roi_type)) + 1
        item = RoiItem(rect, roi_type, f"ROI {count}")
        self.scene().addItem(item)
        item.setSelected(True)
        self.roiChanged.emit()
        return True

    def clear_rois(self, roi_type: str | None = None) -> None:
        for item in tuple(self.scene().items()):
            if isinstance(item, RoiItem) and (roi_type is None or item.roi_type == roi_type):
                self.scene().removeItem(item)
        self.roiChanged.emit()

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
        if self._pixmap.pixmap().isNull():
            super().wheelEvent(event)
            return
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        current_scale = abs(float(self.transform().m11()))
        requested_scale = current_scale * factor
        if not 0.01 <= requested_scale <= 64.0:
            event.accept()
            return
        self._fit = False
        self._last_fit_key = None
        self._set_scrollbars_for_fit(False)
        self.scale(factor, factor)
        self.viewTransformChanged.emit((QtGui.QTransform(self.transform()), self._fit))
        event.accept()

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
            self.viewport().unsetCursor()
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            return
        super().mouseReleaseEvent(event)
