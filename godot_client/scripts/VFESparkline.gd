class_name VFESparkline
extends Control

const MAX_SAMPLES: int = 80
const VFE_THRESHOLD: float = 3.2

var _samples: PackedFloat32Array = PackedFloat32Array()


func _init() -> void:
	custom_minimum_size = Vector2(240.0, 74.0)
	mouse_filter = Control.MOUSE_FILTER_IGNORE


func push_sample(vfe: float) -> void:
	_samples.append(maxf(0.0, vfe))
	if _samples.size() > MAX_SAMPLES:
		_samples = _samples.slice(_samples.size() - MAX_SAMPLES)
	queue_redraw()


func clear_samples() -> void:
	_samples.clear()
	queue_redraw()


func _draw() -> void:
	var w := size.x
	var h := size.y
	if w < 10.0 or h < 10.0:
		return

	var bg_rect := Rect2(Vector2.ZERO, size)
	draw_rect(bg_rect, Color(0.955, 0.968, 0.985, 1.0), true)
	draw_rect(bg_rect, Color(0.82, 0.86, 0.91, 1.0), false, 1.0)

	var pad_left := 8.0
	var pad_right := 8.0
	var pad_top := 18.0
	var pad_bottom := 8.0
	var plot_w := maxf(1.0, w - pad_left - pad_right)
	var plot_h := maxf(1.0, h - pad_top - pad_bottom)

	var max_val := 4.5
	var current_val := 0.0
	for s in _samples:
		max_val = maxf(max_val, s * 1.12)
	if _samples.size() > 0:
		current_val = _samples[_samples.size() - 1]

	# Subtle horizontal grid guides
	var grid_col := Color(0.86, 0.89, 0.93, 0.8)
	for g in range(1, 3):
		var gy := pad_top + plot_h * (float(g) / 3.0)
		draw_line(Vector2(pad_left, gy), Vector2(pad_left + plot_w, gy), grid_col, 1.0)

	# VFE synthesis trigger threshold guide (3.2)
	var thresh_t := clampf(VFE_THRESHOLD / max_val, 0.0, 1.0)
	var thresh_y := pad_top + plot_h * (1.0 - thresh_t)
	var dash_x := pad_left
	var dash_col := Color(0.88, 0.56, 0.16, 0.82)
	while dash_x < pad_left + plot_w:
		var seg_end := minf(dash_x + 5.0, pad_left + plot_w)
		draw_line(Vector2(dash_x, thresh_y), Vector2(seg_end, thresh_y), dash_col, 1.2)
		dash_x += 9.0

	# Draw filled sparkline + crisp polyline
	var count := _samples.size()
	if count >= 2:
		var pts := PackedVector2Array()
		var poly := PackedVector2Array()
		var denom := float(maxi(count - 1, 1))
		for i in range(count):
			var px := pad_left + (float(i) / denom) * plot_w
			var py := pad_top + plot_h * (1.0 - clampf(_samples[i] / max_val, 0.0, 1.0))
			pts.append(Vector2(px, py))
			poly.append(Vector2(px, py))

		poly.append(Vector2(pts[count - 1].x, pad_top + plot_h))
		poly.append(Vector2(pts[0].x, pad_top + plot_h))
		draw_colored_polygon(poly, Color(0.20, 0.52, 0.88, 0.16))

		var line_col := Color(0.82, 0.30, 0.24) if current_val >= VFE_THRESHOLD else Color(0.16, 0.48, 0.86)
		draw_polyline(pts, line_col, 2.0, true)
		draw_circle(pts[count - 1], 3.2, line_col)
	elif count == 1:
		var py := pad_top + plot_h * (1.0 - clampf(_samples[0] / max_val, 0.0, 1.0))
		draw_circle(Vector2(pad_left + plot_w, py), 3.2, Color(0.16, 0.48, 0.86))

	# Compact inline header labels
	var font := ThemeDB.fallback_font
	if font != null:
		draw_string(
			font,
			Vector2(pad_left, 13.0),
			"VFE (Free Energy)",
			HORIZONTAL_ALIGNMENT_LEFT,
			-1.0,
			11,
			Color(0.26, 0.32, 0.40),
		)
		var val_text := "%.2f" % current_val
		var val_col := Color(0.80, 0.26, 0.20) if current_val >= VFE_THRESHOLD else Color(0.14, 0.42, 0.78)
		draw_string(
			font,
			Vector2(w - pad_right - 44.0, 13.0),
			val_text,
			HORIZONTAL_ALIGNMENT_RIGHT,
			44.0,
			11,
			val_col,
		)
