class_name TelemetryHUD
extends CanvasLayer

signal command_requested(payload: Dictionary)
signal follow_toggled(enabled: bool)
signal camera_preset_requested(preset_id: int)
signal archive_refresh_requested()

const COLOR_TEXT := Color(0.14, 0.18, 0.24)
const COLOR_MUTED := Color(0.42, 0.47, 0.55)
const COLOR_BORDER := Color(0.82, 0.86, 0.90)
const COLOR_PANEL := Color(0.985, 0.99, 0.996, 0.94)

var _conn_badge: Label
var _conn_badge_style: StyleBoxFlat
var _llm_badge: Label
var _llm_badge_style: StyleBoxFlat
var _summary_label: Label
var _pause_btn: Button
var _follow_btn: Button
var _collapse_btn: Button
var _speed_buttons: Dictionary = {}  # float hz -> Button

var _side_panel: PanelContainer
var _side_collapsed: bool = false
var _is_paused: bool = false
var _follow_active: bool = false

# Homeostasis bars
var _hom_bars: Dictionary = {}
var _hom_vals: Dictionary = {}
var _temp_label: Label

# PAD & Social meters
var _mood_badge: Label
var _mood_badge_style: StyleBoxFlat
var _pad_bars: Dictionary = {}
var _pad_vals: Dictionary = {}
var _trust_bar: ProgressBar
var _trust_val: Label
var _fear_bar: ProgressBar
var _fear_val: Label
var _calm_bar: ProgressBar
var _calm_val: Label

# VFE sparkline & Skill Library feed
var _sparkline: VFESparkline
var _active_skill_label: Label
var _skill_list_box: VBoxContainer

# Archive spawner
var _archive_select: OptionButton
var _archive_status: Label
var _archive_cell_ids: Array[int] = []

# Prompt input & hover tooltip
var _prompt_input: LineEdit
var _tooltip_panel: PanelContainer
var _tooltip_label: Label

var _last_preset: String = ""
var _last_tick_seen: int = -1
var _last_skill_sig: String = ""


func build_ui() -> void:
	_build_top_bar()
	_build_side_panel()
	_build_bottom_bar()
	_build_hover_tooltip()


func is_text_input_focused() -> bool:
	var owner := get_viewport().gui_get_focus_owner()
	if owner == null:
		return false
	return owner is LineEdit or owner is TextEdit or owner is CodeEdit or owner is SpinBox


func toggle_side_panel() -> void:
	_side_collapsed = not _side_collapsed
	if _side_panel != null:
		_side_panel.visible = not _side_collapsed
	if _collapse_btn != null:
		_collapse_btn.text = "Show HUD" if _side_collapsed else "Hide HUD"


func set_follow_active(enabled: bool) -> void:
	_follow_active = enabled
	if _follow_btn != null:
		_follow_btn.text = "Follow: ON (F)" if enabled else "Follow (F)"
		_style_button(_follow_btn, enabled)


func set_connection_status(is_connected: bool) -> void:
	if _conn_badge == null:
		return
	if is_connected:
		_conn_badge.text = "● Connected"
		_conn_badge.add_theme_color_override("font_color", Color(0.08, 0.48, 0.26))
		_conn_badge_style.bg_color = Color(0.88, 0.97, 0.91)
		_conn_badge_style.border_color = Color(0.62, 0.87, 0.72)
	else:
		_conn_badge.text = "● Offline"
		_conn_badge.add_theme_color_override("font_color", Color(0.72, 0.18, 0.18))
		_conn_badge_style.bg_color = Color(0.99, 0.90, 0.90)
		_conn_badge_style.border_color = Color(0.92, 0.68, 0.68)


func set_archive_elites(elites: Array, metadata: Dictionary) -> void:
	if _archive_select == null:
		return
	_archive_select.clear()
	_archive_cell_ids.clear()

	if elites.is_empty():
		_archive_select.add_item("(No archive elites found)")
		if _archive_status != null:
			_archive_status.text = "Archive empty"
		return

	for raw in elites:
		if not (raw is Dictionary):
			continue
		var item: Dictionary = raw
		var cell_idx: int = int(item.get("cell_index", 0))
		var arch: String = str(item.get("archetype", "elite"))
		var obj_val: float = float(item.get("objective", 0.0))
		_archive_cell_ids.append(cell_idx)
		_archive_select.add_item("#%d %s (%.2f)" % [cell_idx, arch, obj_val])

	if _archive_status != null:
		var cov := float(metadata.get("coverage", 0.0)) * 100.0
		_archive_status.text = "%d elites loaded (%.0f%% cov)" % [_archive_cell_ids.size(), cov]


func set_archive_status(msg: String) -> void:
	if _archive_status != null:
		_archive_status.text = msg


func update_hover_tooltip(meta: Dictionary, screen_pos: Vector2) -> void:
	if _tooltip_panel == null:
		return
	if meta.is_empty():
		_tooltip_panel.visible = false
		return

	var obj_name: String = str(meta.get("name", meta.get("obj_id", "Object")))
	var temp_c: float = float(meta.get("surface_temp_c", 20.0))
	var elev: float = float(meta.get("elevation", 0.0))
	var intact: bool = bool(meta.get("is_intact", true))
	var status_str := "Intact" if intact else "Broken (Reset Room to restore)"
	_tooltip_label.text = "%s\nTemp: %.1f°C  |  Elev: %.2fm\nStatus: %s" % [
		obj_name,
		temp_c,
		elev,
		status_str,
	]
	_tooltip_panel.visible = true
	var vp_size := get_viewport().get_visible_rect().size
	var target_pos := screen_pos + Vector2(16.0, 14.0)
	target_pos.x = minf(target_pos.x, vp_size.x - 240.0)
	target_pos.y = minf(target_pos.y, vp_size.y - 90.0)
	_tooltip_panel.position = target_pos


func update_frame(s: Dictionary) -> void:
	var preset: String = str(s.get("preset_label", "-"))
	var tick: int = int(s.get("tick", 0))
	if preset != _last_preset or tick < _last_tick_seen:
		_last_preset = preset
		_sparkline.clear_samples()
	if tick != _last_tick_seen:
		_last_tick_seen = tick
		_sparkline.push_sample(float(s.get("vfe", 0.0)))

	# Pause & speed button state
	_is_paused = bool(s.get("paused", false))
	_pause_btn.text = "▶ Resume" if _is_paused else "⏸ Pause"
	_style_button(_pause_btn, _is_paused)

	var current_hz: float = float(s.get("tick_hz", 6.0))
	for hz_val in _speed_buttons.keys():
		var btn: Button = _speed_buttons[hz_val]
		_style_button(btn, absf(float(hz_val) - current_hz) < 0.6)

	# Top summary readout
	var mode: String = str(s.get("mode", "active_inference"))
	var action: String = str(s.get("action", "idle"))
	_summary_label.text = "%s  •  Tick %d  •  %s  •  %s" % [preset, tick, mode, action]

	# LLM status badge
	var llm_status: String = str(s.get("llm_status", "idle"))
	var llm_backend: String = str(s.get("llm_backend", "mock"))
	_update_llm_badge(llm_status, llm_backend)

	# Homeostasis bars
	var hom_var: Variant = s.get("homeostasis", {})
	var hom: Dictionary = hom_var if hom_var is Dictionary else {}
	for key in ["warmth", "hunger", "energy", "stimulation", "comfort"]:
		var val: float = clampf(float(hom.get(key, 0.0)), 0.0, 1.0)
		(_hom_bars[key] as ProgressBar).value = val * 100.0
		(_hom_vals[key] as Label).text = "%d%%" % int(round(val * 100.0))

	var core_t: float = float(hom.get("core_temp_c", 38.0))
	var local_t: float = float(s.get("local_temp_c", 20.0))
	_temp_label.text = "Core: %.2f°C   •   Local Env: %.1f°C" % [core_t, local_t]

	# PAD & social meters
	var pad_var: Variant = s.get("pad", {})
	var pad: Dictionary = pad_var if pad_var is Dictionary else {}
	var mood: String = str(pad.get("mood", "neutral_alert"))
	_mood_badge.text = mood.replace("_", " ").capitalize()
	_color_mood_badge(float(pad.get("pleasure", 0.0)), float(pad.get("fear_intensity", 0.0)))

	for pkey in ["pleasure", "arousal", "dominance"]:
		var pv: float = clampf(float(pad.get(pkey, 0.0)), -1.0, 1.0)
		(_pad_bars[pkey] as ProgressBar).value = (pv + 1.0) * 50.0
		(_pad_vals[pkey] as Label).text = "%+.2f" % pv

	var trust: float = clampf(float(s.get("user_trust", 0.0)), -1.0, 1.0)
	var fear: float = clampf(float(s.get("user_fear", 0.0)), 0.0, 1.0)
	var belief_calm: float = clampf(float(s.get("user_belief_calm", 0.5)), 0.0, 1.0)
	_trust_bar.value = (trust + 1.0) * 50.0
	_trust_val.text = "%+.2f" % trust
	_fear_bar.value = fear * 100.0
	_fear_val.text = "%.2f" % fear
	_calm_bar.value = belief_calm * 100.0
	_calm_val.text = "%d%%" % int(round(belief_calm * 100.0))

	# Skill library feed
	var active_skill: Variant = s.get("active_skill_name")
	var active_prim: Variant = s.get("active_primitive")
	var synth_skill: Variant = s.get("synthesized_skill")
	if active_skill != null and not str(active_skill).is_empty():
		var tag := "REUSED" if mode == "skill_reused" else "NEW"
		_active_skill_label.text = "[%s] %s → %s" % [tag, str(active_skill), str(active_prim)]
		_active_skill_label.add_theme_color_override("font_color", Color(0.12, 0.42, 0.78))
	elif synth_skill != null and not str(synth_skill).is_empty():
		_active_skill_label.text = "Synthesized: %s" % str(synth_skill)
		_active_skill_label.add_theme_color_override("font_color", Color(0.10, 0.52, 0.30))
	else:
		_active_skill_label.text = "Active Inference (Fast Path)"
		_active_skill_label.add_theme_color_override("font_color", COLOR_MUTED)

	var skills_var: Variant = s.get("skill_library", [])
	var skills: Array = skills_var if skills_var is Array else []
	var counts_var: Variant = s.get("skill_use_counts", {})
	var counts: Dictionary = counts_var if counts_var is Dictionary else {}
	_rebuild_skill_feed(skills, counts, str(active_skill) if active_skill != null else "")


func _update_llm_badge(status: String, backend: String) -> void:
	_llm_badge.tooltip_text = "LLM Director (%s): %s" % [backend, status]
	if status == "synthesizing":
		_llm_badge.text = "LLM: Synthesizing..."
		_llm_badge.add_theme_color_override("font_color", Color(0.10, 0.38, 0.76))
		_llm_badge_style.bg_color = Color(0.89, 0.94, 1.0)
		_llm_badge_style.border_color = Color(0.64, 0.80, 0.96)
	elif status.begins_with("error"):
		_llm_badge.text = "LLM: Error"
		_llm_badge.add_theme_color_override("font_color", Color(0.74, 0.16, 0.16))
		_llm_badge_style.bg_color = Color(0.99, 0.90, 0.90)
		_llm_badge_style.border_color = Color(0.92, 0.68, 0.68)
	elif backend == "mock":
		_llm_badge.text = "LLM: Fallback"
		_llm_badge.add_theme_color_override("font_color", Color(0.56, 0.38, 0.08))
		_llm_badge_style.bg_color = Color(0.99, 0.96, 0.86)
		_llm_badge_style.border_color = Color(0.90, 0.80, 0.56)
	else:
		_llm_badge.text = "LLM: Idle"
		_llm_badge.add_theme_color_override("font_color", Color(0.18, 0.44, 0.28))
		_llm_badge_style.bg_color = Color(0.91, 0.97, 0.93)
		_llm_badge_style.border_color = Color(0.70, 0.86, 0.75)


func _color_mood_badge(pleasure: float, fear: float) -> void:
	if fear > 0.35:
		_mood_badge.add_theme_color_override("font_color", Color(0.72, 0.18, 0.18))
		_mood_badge_style.bg_color = Color(0.99, 0.91, 0.91)
	elif pleasure >= 0.15:
		_mood_badge.add_theme_color_override("font_color", Color(0.10, 0.48, 0.28))
		_mood_badge_style.bg_color = Color(0.90, 0.97, 0.92)
	else:
		_mood_badge.add_theme_color_override("font_color", Color(0.24, 0.34, 0.48))
		_mood_badge_style.bg_color = Color(0.92, 0.94, 0.97)


func _rebuild_skill_feed(skills: Array, counts: Dictionary, active_name: String) -> void:
	var sig := "%s|%s|%s" % [active_name, JSON.stringify(skills), JSON.stringify(counts)]
	if sig == _last_skill_sig:
		return
	_last_skill_sig = sig

	for child in _skill_list_box.get_children():
		child.queue_free()

	if skills.is_empty():
		var empty_lbl := Label.new()
		empty_lbl.text = "No rituals synthesized yet. Send a prompt or wait for VFE spike."
		empty_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		empty_lbl.add_theme_font_size_override("font_size", 11)
		empty_lbl.add_theme_color_override("font_color", COLOR_MUTED)
		_skill_list_box.add_child(empty_lbl)
		return

	# Show most recent 6 skills
	var start_idx := maxi(0, skills.size() - 6)
	for i in range(skills.size() - 1, start_idx - 1, -1):
		var sname: String = str(skills[i])
		var uses: int = int(counts.get(sname, 1))
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 6)

		var name_lbl := Label.new()
		name_lbl.text = ("▸ " if sname == active_name else "• ") + sname
		name_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		name_lbl.clip_text = true
		name_lbl.add_theme_font_size_override("font_size", 11)
		name_lbl.add_theme_color_override(
			"font_color",
			Color(0.12, 0.42, 0.80) if sname == active_name else COLOR_TEXT,
		)
		row.add_child(name_lbl)

		var tag_lbl := Label.new()
		tag_lbl.text = "REUSED ×%d" % uses if uses > 1 else "NEW"
		tag_lbl.add_theme_font_size_override("font_size", 10)
		tag_lbl.add_theme_color_override(
			"font_color",
			Color(0.10, 0.50, 0.30) if uses > 1 else Color(0.45, 0.38, 0.68),
		)
		row.add_child(tag_lbl)
		_skill_list_box.add_child(row)


func _build_top_bar() -> void:
	var top_panel := PanelContainer.new()
	top_panel.anchor_right = 1.0
	top_panel.offset_bottom = 46.0
	var style := StyleBoxFlat.new()
	style.bg_color = COLOR_PANEL
	style.border_width_bottom = 1
	style.border_color = COLOR_BORDER
	style.content_margin_left = 10
	style.content_margin_right = 10
	style.content_margin_top = 6
	style.content_margin_bottom = 6
	top_panel.add_theme_stylebox_override("panel", style)
	add_child(top_panel)

	var bar := HBoxContainer.new()
	bar.add_theme_constant_override("separation", 8)
	top_panel.add_child(bar)

	var conn_pair := _make_badge("● Connecting...")
	_conn_badge = conn_pair["label"]
	_conn_badge_style = conn_pair["style"]
	bar.add_child(conn_pair["panel"])

	var llm_pair := _make_badge("LLM: Idle")
	_llm_badge = llm_pair["label"]
	_llm_badge_style = llm_pair["style"]
	bar.add_child(llm_pair["panel"])

	_summary_label = Label.new()
	_summary_label.text = "Waiting for PET-AI Bridge..."
	_summary_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_summary_label.clip_text = true
	_summary_label.add_theme_font_size_override("font_size", 12)
	_summary_label.add_theme_color_override("font_color", COLOR_TEXT)
	bar.add_child(_summary_label)

	_pause_btn = _make_button("⏸ Pause", func() -> void:
		command_requested.emit({"type": "set_paused", "paused": not _is_paused})
	)
	bar.add_child(_pause_btn)

	bar.add_child(_make_button("Step", func() -> void:
		command_requested.emit({"type": "step_once"})
	))

	bar.add_child(_make_button("Reset Room", func() -> void:
		command_requested.emit({"type": "reset_room"})
	))

	# Speed selector (0.5x = 3Hz, 1x = 6Hz, 2x = 12Hz, 3x = 18Hz)
	var speeds := [["0.5x", 3.0], ["1x", 6.0], ["2x", 12.0], ["3x", 18.0]]
	for sp in speeds:
		var lbl: String = str(sp[0])
		var hz: float = float(sp[1])
		var sbtn := _make_button(lbl, func() -> void:
			command_requested.emit({"type": "set_speed", "tick_hz": hz})
		)
		_speed_buttons[hz] = sbtn
		bar.add_child(sbtn)

	_follow_btn = _make_button("Follow (F)", func() -> void:
		follow_toggled.emit(not _follow_active)
	)
	bar.add_child(_follow_btn)

	for cam_idx in [1, 2, 3]:
		var idx := int(cam_idx)
		bar.add_child(_make_button("C%d" % idx, func() -> void:
			camera_preset_requested.emit(idx)
		))

	_collapse_btn = _make_button("Hide HUD", func() -> void:
		toggle_side_panel()
	)
	bar.add_child(_collapse_btn)


func _build_side_panel() -> void:
	_side_panel = PanelContainer.new()
	_side_panel.anchor_left = 1.0
	_side_panel.anchor_right = 1.0
	_side_panel.anchor_top = 0.0
	_side_panel.anchor_bottom = 1.0
	_side_panel.offset_left = -316.0
	_side_panel.offset_right = -10.0
	_side_panel.offset_top = 54.0
	_side_panel.offset_bottom = -58.0
	var style := StyleBoxFlat.new()
	style.bg_color = COLOR_PANEL
	style.border_width_left = 1
	style.border_width_right = 1
	style.border_width_top = 1
	style.border_width_bottom = 1
	style.border_color = COLOR_BORDER
	style.corner_radius_top_left = 8
	style.corner_radius_top_right = 8
	style.corner_radius_bottom_left = 8
	style.corner_radius_bottom_right = 8
	style.content_margin_left = 10
	style.content_margin_right = 10
	style.content_margin_top = 10
	style.content_margin_bottom = 10
	_side_panel.add_theme_stylebox_override("panel", style)
	add_child(_side_panel)

	var scroll := ScrollContainer.new()
	scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	_side_panel.add_child(scroll)

	var col := VBoxContainer.new()
	col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	col.add_theme_constant_override("separation", 8)
	scroll.add_child(col)

	# 1. Homeostasis Section
	col.add_child(_make_section_header("HOMEOSTASIS"))
	var hom_defs := [
		["warmth", "Warmth", Color(0.90, 0.48, 0.24)],
		["hunger", "Hunger", Color(0.84, 0.62, 0.20)],
		["energy", "Energy", Color(0.22, 0.64, 0.42)],
		["stimulation", "Stimulation", Color(0.32, 0.52, 0.88)],
		["comfort", "Comfort", Color(0.46, 0.44, 0.80)],
	]
	for hd in hom_defs:
		var k: String = str(hd[0])
		var row := _make_meter_row(str(hd[1]), hd[2] as Color)
		_hom_bars[k] = row["bar"]
		_hom_vals[k] = row["val"]
		col.add_child(row["root"])

	_temp_label = Label.new()
	_temp_label.text = "Core: 38.0°C  •  Local: 20.0°C"
	_temp_label.add_theme_font_size_override("font_size", 11)
	_temp_label.add_theme_color_override("font_color", COLOR_MUTED)
	col.add_child(_temp_label)

	col.add_child(HSeparator.new())

	# 2. PAD Emotions & Social Trust/Fear
	var pad_hdr := HBoxContainer.new()
	var pad_title := _make_section_header("EMOTION (PAD) & SOCIAL")
	pad_title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	pad_hdr.add_child(pad_title)
	var mood_pair := _make_badge("Neutral Alert")
	_mood_badge = mood_pair["label"]
	_mood_badge_style = mood_pair["style"]
	pad_hdr.add_child(mood_pair["panel"])
	col.add_child(pad_hdr)

	for pd in [
		["pleasure", "Pleasure", Color(0.20, 0.64, 0.46)],
		["arousal", "Arousal", Color(0.88, 0.46, 0.24)],
		["dominance", "Dominance", Color(0.40, 0.46, 0.84)],
	]:
		var pk: String = str(pd[0])
		var prow := _make_meter_row(str(pd[1]), pd[2] as Color)
		_pad_bars[pk] = prow["bar"]
		_pad_vals[pk] = prow["val"]
		col.add_child(prow["root"])

	var trust_row := _make_meter_row("User Trust", Color(0.18, 0.56, 0.78))
	_trust_bar = trust_row["bar"]
	_trust_val = trust_row["val"]
	col.add_child(trust_row["root"])

	var fear_row := _make_meter_row("User Fear", Color(0.84, 0.30, 0.30))
	_fear_bar = fear_row["bar"]
	_fear_val = fear_row["val"]
	col.add_child(fear_row["root"])

	var calm_row := _make_meter_row("Calm Belief", Color(0.32, 0.62, 0.58))
	_calm_bar = calm_row["bar"]
	_calm_val = calm_row["val"]
	col.add_child(calm_row["root"])

	col.add_child(HSeparator.new())

	# 3. VFE Sparkline
	_sparkline = VFESparkline.new()
	_sparkline.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	col.add_child(_sparkline)

	col.add_child(HSeparator.new())

	# 4. Skill Library Feed
	col.add_child(_make_section_header("SKILL LIBRARY"))
	_active_skill_label = Label.new()
	_active_skill_label.text = "Active Inference (Fast Path)"
	_active_skill_label.clip_text = true
	_active_skill_label.add_theme_font_size_override("font_size", 11)
	_active_skill_label.add_theme_color_override("font_color", COLOR_MUTED)
	col.add_child(_active_skill_label)

	_skill_list_box = VBoxContainer.new()
	_skill_list_box.add_theme_constant_override("separation", 3)
	col.add_child(_skill_list_box)

	col.add_child(HSeparator.new())

	# 5. Presets & CMA-MAE Archive Spawner
	col.add_child(_make_section_header("GENOME PRESETS & ARCHIVE"))
	var preset_grid := GridContainer.new()
	preset_grid.columns = 2
	preset_grid.add_theme_constant_override("h_separation", 6)
	preset_grid.add_theme_constant_override("v_separation", 5)
	for pname in [
		"impossible_cat",
		"hairless_sphynx",
		"staring_observer",
		"chaotic_gremlin",
		"shy_hermit",
	]:
		var p_id := str(pname)
		var btn := _make_button(p_id.replace("_", " ").capitalize(), func() -> void:
			command_requested.emit({"type": "spawn_preset", "preset": p_id})
		)
		btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		preset_grid.add_child(btn)
	col.add_child(preset_grid)

	var arch_row := HBoxContainer.new()
	arch_row.add_theme_constant_override("separation", 5)
	_archive_select = OptionButton.new()
	_archive_select.focus_mode = Control.FOCUS_NONE
	_archive_select.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_archive_select.clip_text = true
	_archive_select.add_item("Loading archive...")
	_style_button(_archive_select, false)
	arch_row.add_child(_archive_select)

	arch_row.add_child(_make_button("Spawn", func() -> void:
		var sel := _archive_select.selected
		if sel >= 0 and sel < _archive_cell_ids.size():
			command_requested.emit({
				"type": "spawn_archive_cell",
				"cell_index": _archive_cell_ids[sel],
			})
	))
	arch_row.add_child(_make_button("↻", func() -> void:
		archive_refresh_requested.emit()
	))
	col.add_child(arch_row)

	_archive_status = Label.new()
	_archive_status.text = "Fetching CMA-MAE archive..."
	_archive_status.add_theme_font_size_override("font_size", 10)
	_archive_status.add_theme_color_override("font_color", COLOR_MUTED)
	col.add_child(_archive_status)

	col.add_child(HSeparator.new())

	# 6. Sensory Events
	col.add_child(_make_section_header("SENSORY EVENTS"))
	var ev_grid := GridContainer.new()
	ev_grid.columns = 2
	ev_grid.add_theme_constant_override("h_separation", 6)
	ev_grid.add_theme_constant_override("v_separation", 5)
	for ev in [
		["Pet Gently", "calm_petting"],
		["Feed", "gentle_feeding"],
		["Loud Clatter", "loud_clatter"],
		["Shout", "startling_shout"],
	]:
		var ev_label: String = str(ev[0])
		var ev_id: String = str(ev[1])
		var ebtn := _make_button(ev_label, func() -> void:
			command_requested.emit({"type": "sensory_event", "event": ev_id})
		)
		ebtn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		ev_grid.add_child(ebtn)
	col.add_child(ev_grid)


func _build_bottom_bar() -> void:
	var bottom_panel := PanelContainer.new()
	bottom_panel.anchor_top = 1.0
	bottom_panel.anchor_bottom = 1.0
	bottom_panel.anchor_right = 1.0
	bottom_panel.offset_top = -50.0
	bottom_panel.offset_left = 10.0
	bottom_panel.offset_right = -10.0
	bottom_panel.offset_bottom = -8.0
	var style := StyleBoxFlat.new()
	style.bg_color = COLOR_PANEL
	style.border_width_left = 1
	style.border_width_right = 1
	style.border_width_top = 1
	style.border_width_bottom = 1
	style.border_color = COLOR_BORDER
	style.corner_radius_top_left = 8
	style.corner_radius_top_right = 8
	style.corner_radius_bottom_left = 8
	style.corner_radius_bottom_right = 8
	style.content_margin_left = 10
	style.content_margin_right = 10
	style.content_margin_top = 5
	style.content_margin_bottom = 5
	bottom_panel.add_theme_stylebox_override("panel", style)
	add_child(bottom_panel)

	var hbox := HBoxContainer.new()
	hbox.add_theme_constant_override("separation", 8)
	bottom_panel.add_child(hbox)

	_prompt_input = LineEdit.new()
	_prompt_input.placeholder_text = "Send prompt to prefrontal director (e.g. ceiling high_shelf gravity ritual)..."
	_prompt_input.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_prompt_input.add_theme_font_size_override("font_size", 12)
	_prompt_input.add_theme_color_override("font_color", COLOR_TEXT)
	_prompt_input.add_theme_color_override("font_placeholder_color", COLOR_MUTED)
	_prompt_input.add_theme_color_override("caret_color", COLOR_TEXT)
	var le_style := StyleBoxFlat.new()
	le_style.bg_color = Color(0.95, 0.96, 0.98)
	le_style.border_width_left = 1
	le_style.border_width_right = 1
	le_style.border_width_top = 1
	le_style.border_width_bottom = 1
	le_style.border_color = COLOR_BORDER
	le_style.corner_radius_top_left = 5
	le_style.corner_radius_top_right = 5
	le_style.corner_radius_bottom_left = 5
	le_style.corner_radius_bottom_right = 5
	le_style.content_margin_left = 8
	le_style.content_margin_right = 8
	_prompt_input.add_theme_stylebox_override("normal", le_style)
	_prompt_input.add_theme_stylebox_override("focus", le_style)
	_prompt_input.text_submitted.connect(_on_prompt_submitted)
	hbox.add_child(_prompt_input)

	hbox.add_child(_make_button("Send", func() -> void:
		_on_prompt_submitted(_prompt_input.text)
	))

	var hint := Label.new()
	hint.text = "LMB Move Human • RMB Orbit • Wheel Zoom • WASD Pan • F Follow • 1/2/3 Cam • Space Pause"
	hint.add_theme_font_size_override("font_size", 10)
	hint.add_theme_color_override("font_color", COLOR_MUTED)
	hbox.add_child(hint)


func _build_hover_tooltip() -> void:
	_tooltip_panel = PanelContainer.new()
	_tooltip_panel.visible = false
	_tooltip_panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.99, 0.995, 1.0, 0.96)
	style.border_width_left = 1
	style.border_width_right = 1
	style.border_width_top = 1
	style.border_width_bottom = 1
	style.border_color = Color(0.68, 0.76, 0.86)
	style.corner_radius_top_left = 6
	style.corner_radius_top_right = 6
	style.corner_radius_bottom_left = 6
	style.corner_radius_bottom_right = 6
	style.content_margin_left = 9
	style.content_margin_right = 9
	style.content_margin_top = 6
	style.content_margin_bottom = 6
	_tooltip_panel.add_theme_stylebox_override("panel", style)
	add_child(_tooltip_panel)

	_tooltip_label = Label.new()
	_tooltip_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_tooltip_label.add_theme_font_size_override("font_size", 11)
	_tooltip_label.add_theme_color_override("font_color", COLOR_TEXT)
	_tooltip_panel.add_child(_tooltip_label)


func _on_prompt_submitted(text: String) -> void:
	var trimmed := text.strip_edges()
	_prompt_input.release_focus()
	if trimmed.is_empty():
		return
	command_requested.emit({"type": "user_prompt", "text": trimmed})
	_prompt_input.clear()


func _make_section_header(title: String) -> Label:
	var lbl := Label.new()
	lbl.text = title
	lbl.add_theme_font_size_override("font_size", 10)
	lbl.add_theme_color_override("font_color", COLOR_MUTED)
	return lbl


func _make_badge(text: String) -> Dictionary:
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.93, 0.95, 0.98)
	style.border_width_left = 1
	style.border_width_right = 1
	style.border_width_top = 1
	style.border_width_bottom = 1
	style.border_color = COLOR_BORDER
	style.corner_radius_top_left = 10
	style.corner_radius_top_right = 10
	style.corner_radius_bottom_left = 10
	style.corner_radius_bottom_right = 10
	style.content_margin_left = 8
	style.content_margin_right = 8
	style.content_margin_top = 2
	style.content_margin_bottom = 2
	panel.add_theme_stylebox_override("panel", style)

	var lbl := Label.new()
	lbl.text = text
	lbl.add_theme_font_size_override("font_size", 11)
	lbl.add_theme_color_override("font_color", COLOR_TEXT)
	panel.add_child(lbl)
	return {"panel": panel, "label": lbl, "style": style}


func _make_meter_row(label_text: String, fill_color: Color) -> Dictionary:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 6)

	var lbl := Label.new()
	lbl.text = label_text
	lbl.custom_minimum_size = Vector2(78.0, 0.0)
	lbl.add_theme_font_size_override("font_size", 11)
	lbl.add_theme_color_override("font_color", COLOR_TEXT)
	row.add_child(lbl)

	var bar := ProgressBar.new()
	bar.min_value = 0.0
	bar.max_value = 100.0
	bar.value = 50.0
	bar.show_percentage = false
	bar.custom_minimum_size = Vector2(0.0, 12.0)
	bar.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	bar.size_flags_vertical = Control.SIZE_SHRINK_CENTER

	var bg := StyleBoxFlat.new()
	bg.bg_color = Color(0.89, 0.91, 0.94)
	bg.corner_radius_top_left = 4
	bg.corner_radius_top_right = 4
	bg.corner_radius_bottom_left = 4
	bg.corner_radius_bottom_right = 4
	bar.add_theme_stylebox_override("background", bg)

	var fill := StyleBoxFlat.new()
	fill.bg_color = fill_color
	fill.corner_radius_top_left = 4
	fill.corner_radius_top_right = 4
	fill.corner_radius_bottom_left = 4
	fill.corner_radius_bottom_right = 4
	bar.add_theme_stylebox_override("fill", fill)
	row.add_child(bar)

	var val := Label.new()
	val.text = "0%"
	val.custom_minimum_size = Vector2(38.0, 0.0)
	val.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	val.add_theme_font_size_override("font_size", 11)
	val.add_theme_color_override("font_color", COLOR_MUTED)
	row.add_child(val)

	return {"root": row, "bar": bar, "val": val}


func _make_button(text: String, on_pressed: Callable) -> Button:
	var btn := Button.new()
	btn.text = text
	btn.focus_mode = Control.FOCUS_NONE
	btn.pressed.connect(on_pressed)
	_style_button(btn, false)
	return btn


func _style_button(btn: Button, active: bool) -> void:
	btn.add_theme_font_size_override("font_size", 11)
	btn.add_theme_color_override(
		"font_color",
		Color(0.09, 0.34, 0.72) if active else COLOR_TEXT,
	)
	btn.add_theme_color_override("font_hover_color", Color(0.08, 0.30, 0.68))
	btn.add_theme_color_override("font_pressed_color", Color(0.06, 0.24, 0.56))

	var normal := StyleBoxFlat.new()
	normal.bg_color = Color(0.87, 0.93, 1.0) if active else Color(0.95, 0.96, 0.98)
	normal.border_width_left = 1
	normal.border_width_right = 1
	normal.border_width_top = 1
	normal.border_width_bottom = 1
	normal.border_color = Color(0.56, 0.74, 0.94) if active else COLOR_BORDER
	normal.corner_radius_top_left = 5
	normal.corner_radius_top_right = 5
	normal.corner_radius_bottom_left = 5
	normal.corner_radius_bottom_right = 5
	normal.content_margin_left = 8
	normal.content_margin_right = 8
	normal.content_margin_top = 4
	normal.content_margin_bottom = 4

	var hover := normal.duplicate() as StyleBoxFlat
	hover.bg_color = Color(0.84, 0.91, 0.99)

	btn.add_theme_stylebox_override("normal", normal)
	btn.add_theme_stylebox_override("hover", hover)
	btn.add_theme_stylebox_override("pressed", hover)
