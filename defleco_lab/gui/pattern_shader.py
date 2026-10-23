"""Native-pixel OpenGL pattern rendering (the owning context must be current)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PySide6 import QtGui, QtOpenGL

if TYPE_CHECKING:
    from .pattern_output import PatternSettings


_VERTEX = """#version 330 core
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_FRAGMENT = """#version 330 core
uniform vec2 dimensions;
uniform vec2 centerShift;
uniform vec2 tilt;
uniform float period;
uniform float angle;
uniform float phase;
uniform float duty;
uniform float powerValue;
uniform float spiralWidth;
uniform float arms;
uniform float spokes;
uniform float peak;
uniform int family;
uniform bool sinusoidal;
uniform bool inverted;
uniform bool antialias;
uniform sampler2D speckle;
out vec4 color;
const float TAU = 6.283185307179586;

float binaryCarrier(float cycles, float width, bool line, float footprint) {
    float p = fract(cycles);
    if (!antialias) {
        if (line) return float(min(p, 1.0-p) <= width * 0.5);
        return float(p < width);
    }
    // Periodic signed distance avoids an artificial edge at the phase wrap.
    float shifted = cycles - (line ? 0.0 : width * 0.5);
    float distance = abs(fract(shifted + 0.5) - 0.5);
    float pixelWidth = max(footprint, 0.000001);
    return clamp((width * 0.5 - distance) / pixelWidth + 0.5, 0.0, 1.0);
}
float binaryCarrier(float cycles, float width, bool line) {
    return binaryCarrier(cycles, width, line, fwidth(cycles));
}
float carrier(float cycles) {
    if (sinusoidal) return (sin(TAU * cycles) + 1.0) * 0.5;
    return binaryCarrier(cycles, duty, false);
}
void main() {
    vec2 pixel = vec2(gl_FragCoord.x - 0.5,
                      dimensions.y - gl_FragCoord.y - 0.5);
    if (family == 9) {
        float value = texture(speckle, (pixel + 0.5) / dimensions).r;
        color = vec4(value, value, value, 1.0);
        return;
    }
    vec2 xy = pixel - (dimensions - 1.0) * 0.5 - dimensions * centerShift;
    float localPhase = phase + dot(tilt, xy / max(dimensions - 1.0, vec2(1.0)));
    float c = cos(angle), s = sin(angle);
    vec2 rotated = vec2(c*xy.x + s*xy.y, -s*xy.x + c*xy.y);
    float radius = length(xy);
    float theta = radius == 0.0 ? 0.0 : atan(xy.y, xy.x);
    // Analytic derivatives stay continuous across atan's -pi/+pi seam.
    vec2 angularGradient = vec2(-xy.y, xy.x) / max(dot(xy, xy), 0.25);
    vec2 radialGradient = xy / max(radius, 0.000001) / period;
    vec2 phaseGradient = tilt / max(dimensions - 1.0, vec2(1.0));
    float signal;
    if (family == 1) {
        float a = binaryCarrier((rotated.x / period + localPhase) * 0.5, 0.5, false);
        float b = binaryCarrier((rotated.y / period + localPhase) * 0.5, 0.5, false);
        signal = a*b + (1.0-a)*(1.0-b);
    } else if (family == 2) {
        signal = carrier(radius / period + localPhase);
    } else if (family == 3) {
        signal = 0.5 * (carrier(rotated.x / period + localPhase)
                      + carrier(rotated.y / period + localPhase));
    } else if (family == 4 || family == 5) {
        vec2 a = abs(rotated);
        float scale = max(a.x, a.y);
        float metric = scale;
        if (family == 5 && scale > 0.0) {
            vec2 normalized = a / scale;
            metric *= pow(pow(normalized.x, powerValue)
                        + pow(normalized.y, powerValue), 1.0 / powerValue);
        }
        signal = binaryCarrier(metric / period + localPhase, duty, true);
    } else if (family == 6 || family == 7) {
        float cycles = radius / period - arms * (theta-angle) / TAU;
        vec2 gradient = radialGradient - arms / TAU * angularGradient + phaseGradient;
        signal = binaryCarrier(cycles + localPhase, spiralWidth, true,
                               abs(gradient.x) + abs(gradient.y));
        if (family == 7) {
            float opposite = radius / period + arms * (theta+angle) / TAU;
            vec2 oppositeGradient = radialGradient + arms / TAU * angularGradient
                                    + phaseGradient;
            signal = max(signal, binaryCarrier(opposite + localPhase, spiralWidth, true,
                         abs(oppositeGradient.x) + abs(oppositeGradient.y)));
        }
    } else if (family == 8) {
        vec2 gradient = spokes / TAU * angularGradient + phaseGradient * 0.5;
        signal = binaryCarrier((theta-angle) * spokes / TAU + localPhase * 0.5,
                               0.5, false, abs(gradient.x) + abs(gradient.y));
    } else if (family == 10) {
        signal = 1.0;
    } else {
        signal = carrier(rotated.x / period + localPhase);
    }
    if (inverted) signal = 1.0 - signal;
    float value = roundEven(clamp(signal * peak, 0.0, 255.0)) / 255.0;
    color = vec4(value, value, value, 1.0);
}
"""

_FAMILIES = {
    name: index for index, name in enumerate(
        ("stripes", "checker", "rings", "composite", "nested_square", "squircle",
         "spiral", "counter_spiral", "starburst", "speckle", "solid")
    )
}


class PatternShader:
    """A full-screen triangle and procedural fragment shader, with cached speckle."""

    def __init__(self) -> None:
        self._context: QtGui.QOpenGLContext | None = None
        self._program: QtOpenGL.QOpenGLShaderProgram | None = None
        self._vao: QtOpenGL.QOpenGLVertexArrayObject | None = None
        self._texture: QtOpenGL.QOpenGLTexture | None = None
        self._speckle_key: tuple | None = None
        self._uniform_locations: dict[str, int] = {}

    def initialize(self, context: QtGui.QOpenGLContext) -> None:
        """Compile in the supplied current OpenGL 3.3 context."""
        self.close()
        self._context = context
        program = QtOpenGL.QOpenGLShaderProgram()
        self._program = program
        for kind, source in (
            (QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex, _VERTEX),
            (QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment, _FRAGMENT),
        ):
            if not program.addShaderFromSourceCode(kind, source):
                raise RuntimeError(f"Pattern shader compilation failed: {program.log()}")
        if not program.link():
            raise RuntimeError(f"Pattern shader link failed: {program.log()}")
        self._uniform_locations = {
            name: program.uniformLocation(name)
            for name in (
                "dimensions", "centerShift", "tilt", "period", "angle", "phase",
                "duty", "powerValue", "spiralWidth", "arms", "spokes", "peak",
                "family", "sinusoidal", "inverted", "antialias", "speckle",
            )
        }
        self._vao = QtOpenGL.QOpenGLVertexArrayObject()
        if not self._vao.create():
            raise RuntimeError("Cannot create pattern shader vertex array")

    def draw(self, settings: PatternSettings, width: int, height: int,
             antialias: bool = True) -> None:
        """Draw to the current framebuffer; dimensions are physical display pixels."""
        if self._program is None or self._vao is None or self._context is None:
            raise RuntimeError("PatternShader has not been initialized")
        width, height = max(2, int(width)), max(2, int(height))
        if settings.family == "speckle":
            key = (width, height, settings)
            if key != self._speckle_key:
                # Preserve numpy's established seeded RNG and Mono8 rounding.
                from .pattern_output import render_pattern

                frame = render_pattern(width, height, settings)
                image = QtGui.QImage(frame.data, width, height, frame.strides[0],
                                     QtGui.QImage.Format.Format_Grayscale8).copy()
                if self._texture is not None:
                    self._texture.destroy()
                self._texture = QtOpenGL.QOpenGLTexture(image)
                self._texture.setMinMagFilters(QtOpenGL.QOpenGLTexture.Filter.Nearest,
                                               QtOpenGL.QOpenGLTexture.Filter.Nearest)
                self._texture.setWrapMode(QtOpenGL.QOpenGLTexture.WrapMode.ClampToEdge)
                self._speckle_key = key
        gl = self._context.functions()
        gl.glViewport(0, 0, width, height)
        gl.glDisable(0x0B71)  # GL_DEPTH_TEST
        gl.glDisable(0x0BE2)  # GL_BLEND
        gl.glDisable(0x0BD0)  # GL_DITHER: preserve explicit Mono8 quantization.
        gl.glDisable(0x8DB9)  # GL_FRAMEBUFFER_SRGB
        program = self._program
        program.bind()
        uniforms = {
            "dimensions": QtGui.QVector2D(width, height),
            "centerShift": QtGui.QVector2D(settings.center_x_percent / 100.0,
                                           settings.center_y_percent / 100.0),
            "tilt": QtGui.QVector2D(settings.phase_tilt_x_deg / 360.0,
                                    settings.phase_tilt_y_deg / 360.0),
            "period": max(2.0, float(settings.period_px)),
            "angle": math.radians(settings.angle_deg),
            "phase": (settings.phase_deg % (720.0 if settings.family == "starburst"
                                            else 360.0)) / 360.0,
            "duty": min(0.99, max(0.01, settings.duty_percent / 100.0)),
            "powerValue": max(2.0, float(settings.squircle_power)),
            "spiralWidth": min(0.9, max(0.02, settings.spiral_width_percent / 100.0)),
            "arms": float(max(1, int(settings.spiral_arms))),
            "spokes": float(max(2, int(settings.spokes))),
            "peak": float(round(min(100.0, max(0.0, settings.brightness)) * 255 / 100)),
            "family": _FAMILIES.get(settings.family, 0),
            "sinusoidal": settings.waveform == "sinusoidal",
            "inverted": settings.invert,
            "antialias": antialias,
            "speckle": 0,
        }
        for name, value in uniforms.items():
            location = self._uniform_locations[name]
            if isinstance(value, QtGui.QVector2D):
                program.setUniformValue(location, value)
            elif isinstance(value, (bool, int)):
                program.setUniformValue1i(location, int(value))
            else:
                program.setUniformValue1f(location, float(value))
        if settings.family == "speckle" and self._texture is not None:
            self._texture.bind(0)
        self._vao.bind()
        gl.glDrawArrays(0x0004, 0, 3)  # GL_TRIANGLES
        self._vao.release()
        if settings.family == "speckle" and self._texture is not None:
            self._texture.release(0)
        program.release()

    def close(self) -> None:
        """Release GPU resources while their owning context is current."""
        if self._texture is not None:
            self._texture.destroy()
            self._texture = None
        self._speckle_key = None
        if self._vao is not None:
            self._vao.destroy()
            self._vao = None
        if self._program is not None:
            self._program.removeAllShaders()
            self._program = None
        self._context = None
        self._uniform_locations.clear()
