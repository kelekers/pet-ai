class_name ProceduralPet
extends Node3D

const TORSO_BASE_HEIGHT: float = 0.24

var flip_pivot: Node3D
var torso_pivot: Node3D
var head_pivot: Node3D
var left_ear: Node3D
var right_ear: Node3D
var left_pupil: MeshInstance3D
var right_pupil: MeshInstance3D
var left_eye_root: Node3D
var right_eye_root: Node3D
var tail_segments: Array[Node3D] = []
var leg_pivots: Array[Node3D] = []

var _body_mat: StandardMaterial3D
var _accent_mat: StandardMaterial3D
var _fur_shell_mat: StandardMaterial3D
var _fur_shell_mesh: MeshInstance3D
var _collar_mat: StandardMaterial3D
var _shadow_disc: MeshInstance3D
var _shadow_mat: StandardMaterial3D
var _speech_label: Label3D
var _telepathy_beam: MeshInstance3D

var _target_pos: Vector3 = Vector3.ZERO
var _gravity_inverted: bool = false
var _phased_active: bool = false
var _body_scale: float = 1.0
var _last_speed: float = 0.0
var _posture_state: String = "sitting_alert"
var _telepathy_target_id: String = ""
var _room_ref: RoomBuilder = null
var _has_received_state: bool = false


func build_rig() -> void:
	_body_mat = StandardMaterial3D.new()
	_body_mat.albedo_color = Color(0.86, 0.72, 0.64)
	_body_mat.roughness = 0.66

	_accent_mat = StandardMaterial3D.new()
	_accent_mat.albedo_color = Color(0.92, 0.80, 0.74)
	_accent_mat.roughness = 0.62

	_fur_shell_mat = StandardMaterial3D.new()
	_fur_shell_mat.albedo_color = Color(0.34, 0.38, 0.44, 0.90)
	_fur_shell_mat.roughness = 0.94
	_fur_shell_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA

	_collar_mat = StandardMaterial3D.new()
	_collar_mat.albedo_color = Color(0.76, 0.28, 0.26)
	_collar_mat.roughness = 0.50

	# Soft contact shadow disc (stays flat on floor or ceiling independent of torso flip)
	_shadow_disc = MeshInstance3D.new()
	var sh_cyl := CylinderMesh.new()
	sh_cyl.top_radius = 0.24
	sh_cyl.bottom_radius = 0.24
	sh_cyl.height = 0.004
	_shadow_disc.mesh = sh_cyl
	_shadow_mat = StandardMaterial3D.new()
	_shadow_mat.albedo_color = Color(0.08, 0.10, 0.14, 0.24)
	_shadow_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	_shadow_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	_shadow_disc.material_override = _shadow_mat
	_shadow_disc.position = Vector3(0.0, 0.006, 0.0)
	_shadow_disc.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(_shadow_disc)

	# Flip pivot sits at torso center height so gravity inversion rotates cleanly in-place
	flip_pivot = Node3D.new()
	flip_pivot.name = "FlipPivot"
	flip_pivot.position = Vector3(0.0, TORSO_BASE_HEIGHT, 0.0)
	add_child(flip_pivot)

	# Torso pivot (local origin at flip_pivot)
	torso_pivot = Node3D.new()
	torso_pivot.name = "TorsoPivot"
	flip_pivot.add_child(torso_pivot)

	# Core torso capsule + subtle chest/haunch sculpting
	var body_mi := MeshInstance3D.new()
	var body_cap := CapsuleMesh.new()
	body_cap.radius = 0.125
	body_cap.height = 0.47
	body_mi.mesh = body_cap
	body_mi.rotation.x = PI * 0.5
	body_mi.material_override = _body_mat
	torso_pivot.add_child(body_mi)

	var chest_mi := MeshInstance3D.new()
	var chest_sph := SphereMesh.new()
	chest_sph.radius = 0.112
	chest_sph.height = 0.21
	chest_mi.mesh = chest_sph
	chest_mi.position = Vector3(0.0, 0.015, 0.14)
	chest_mi.material_override = _body_mat
	torso_pivot.add_child(chest_mi)

	# Procedural fur shell (thickness and alpha scale directly from genome.fur_density)
	_fur_shell_mesh = MeshInstance3D.new()
	var fur_cap := CapsuleMesh.new()
	fur_cap.radius = 0.148
	fur_cap.height = 0.50
	_fur_shell_mesh.mesh = fur_cap
	_fur_shell_mesh.rotation.x = PI * 0.5
	_fur_shell_mesh.material_override = _fur_shell_mat
	torso_pivot.add_child(_fur_shell_mesh)

	# Collar + metallic tag around neck
	var collar := MeshInstance3D.new()
	var col_torus := TorusMesh.new()
	col_torus.inner_radius = 0.078
	col_torus.outer_radius = 0.095
	collar.mesh = col_torus
	collar.position = Vector3(0.0, 0.068, 0.205)
	collar.rotation.x = deg_to_rad(56.0)
	collar.material_override = _collar_mat
	torso_pivot.add_child(collar)

	var tag := MeshInstance3D.new()
	var tag_cyl := CylinderMesh.new()
	tag_cyl.top_radius = 0.019
	tag_cyl.bottom_radius = 0.019
	tag_cyl.height = 0.008
	tag.mesh = tag_cyl
	var tag_mat := StandardMaterial3D.new()
	tag_mat.albedo_color = Color(0.92, 0.78, 0.34)
	tag_mat.metallic = 0.75
	tag_mat.roughness = 0.28
	tag.material_override = tag_mat
	tag.position = Vector3(0.0, 0.005, 0.268)
	tag.rotation.x = PI * 0.5
	torso_pivot.add_child(tag)

	# Head pivot + cranium + muzzle + pink nose + whiskers + ears + eyes
	head_pivot = Node3D.new()
	head_pivot.name = "HeadPivot"
	head_pivot.position = Vector3(0.0, 0.105, 0.245)
	torso_pivot.add_child(head_pivot)

	var head_mi := MeshInstance3D.new()
	var head_sph := SphereMesh.new()
	head_sph.radius = 0.112
	head_sph.height = 0.205
	head_mi.mesh = head_sph
	head_mi.material_override = _body_mat
	head_pivot.add_child(head_mi)

	# Snout / muzzle cheeks
	var muzzle_l := MeshInstance3D.new()
	var muz_sph := SphereMesh.new()
	muz_sph.radius = 0.042
	muz_sph.height = 0.068
	muzzle_l.mesh = muz_sph
	muzzle_l.position = Vector3(-0.025, -0.028, 0.092)
	muzzle_l.material_override = _accent_mat
	head_pivot.add_child(muzzle_l)

	var muzzle_r := MeshInstance3D.new()
	muzzle_r.mesh = muz_sph
	muzzle_r.position = Vector3(0.025, -0.028, 0.092)
	muzzle_r.material_override = _accent_mat
	head_pivot.add_child(muzzle_r)

	# Pink nose
	var nose := MeshInstance3D.new()
	var nose_sph := SphereMesh.new()
	nose_sph.radius = 0.017
	nose_sph.height = 0.024
	nose.mesh = nose_sph
	var nose_mat := StandardMaterial3D.new()
	nose_mat.albedo_color = Color(0.92, 0.58, 0.62)
	nose_mat.roughness = 0.42
	nose.material_override = nose_mat
	nose.position = Vector3(0.0, -0.012, 0.122)
	head_pivot.add_child(nose)

	# Fine whiskers (3 per side)
	var whisker_mat := StandardMaterial3D.new()
	whisker_mat.albedo_color = Color(0.88, 0.89, 0.91)
	whisker_mat.roughness = 0.5
	for side in [-1.0, 1.0]:
		for w_idx in range(-1, 2):
			var w := MeshInstance3D.new()
			var w_box := BoxMesh.new()
			w_box.size = Vector3(0.095, 0.003, 0.003)
			w.mesh = w_box
			w.material_override = whisker_mat
			w.position = Vector3(side * 0.082, -0.026 + float(w_idx) * 0.012, 0.096)
			w.rotation.z = side * float(w_idx) * 0.14
			w.rotation.y = -side * 0.18
			w.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			head_pivot.add_child(w)

	left_ear = _make_ear(Vector3(-0.068, 0.088, -0.006), 0.18)
	right_ear = _make_ear(Vector3(0.068, 0.088, -0.006), -0.18)
	head_pivot.add_child(left_ear)
	head_pivot.add_child(right_ear)

	var left_eye_pair := _make_eye(Vector3(-0.046, 0.022, 0.092))
	left_eye_root = left_eye_pair["root"]
	left_pupil = left_eye_pair["pupil"]

	var right_eye_pair := _make_eye(Vector3(0.046, 0.022, 0.092))
	right_eye_root = right_eye_pair["root"]
	right_pupil = right_eye_pair["pupil"]

	# 4 articulated legs with distinct paws
	var leg_offsets: Array[Vector3] = [
		Vector3(-0.078, -0.035, 0.145),   # Front-Left
		Vector3(0.078, -0.035, 0.145),    # Front-Right
		Vector3(-0.082, -0.035, -0.145),  # Hind-Left
		Vector3(0.082, -0.035, -0.145),   # Hind-Right
	]
	for idx in range(leg_offsets.size()):
		var pivot := Node3D.new()
		pivot.position = leg_offsets[idx]
		var leg_mi := MeshInstance3D.new()
		var leg_cap := CapsuleMesh.new()
		leg_cap.radius = 0.030 if idx < 2 else 0.035
		leg_cap.height = 0.185
		leg_mi.mesh = leg_cap
		leg_mi.position = Vector3(0.0, -0.092, 0.0)
		leg_mi.material_override = _body_mat
		pivot.add_child(leg_mi)

		# Rounded paw at foot tip (reaches y = -0.205 from hip -> -0.24 from torso center)
		var paw_mi := MeshInstance3D.new()
		var paw_sph := SphereMesh.new()
		paw_sph.radius = 0.034
		paw_sph.height = 0.046
		paw_mi.mesh = paw_sph
		paw_mi.position = Vector3(0.0, -0.184, 0.022)
		paw_mi.material_override = _accent_mat
		pivot.add_child(paw_mi)

		torso_pivot.add_child(pivot)
		leg_pivots.append(pivot)

	# 5-segment smoothly tapered procedural tail chain
	var parent_tail: Node3D = torso_pivot
	for i in range(5):
		var seg := Node3D.new()
		seg.position = Vector3(0.0, 0.045, -0.215) if i == 0 else Vector3(0.0, 0.0, -0.082)
		var seg_mi := MeshInstance3D.new()
		var seg_cap := CapsuleMesh.new()
		seg_cap.radius = maxf(0.011, 0.025 - float(i) * 0.0032)
		seg_cap.height = 0.096
		seg_mi.mesh = seg_cap
		seg_mi.rotation.x = PI * 0.5
		seg_mi.position = Vector3(0.0, 0.0, -0.042)
		seg_mi.material_override = _body_mat
		seg.add_child(seg_mi)
		parent_tail.add_child(seg)
		tail_segments.append(seg)
		parent_tail = seg

	# Floating speech / ritual badge (attached to upright root, never upside-down)
	_speech_label = Label3D.new()
	_speech_label.font_size = 30
	_speech_label.pixel_size = 0.0038
	_speech_label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	_speech_label.modulate = Color(0.12, 0.16, 0.22)
	_speech_label.outline_size = 8
	_speech_label.outline_modulate = Color(1.0, 1.0, 1.0, 0.95)
	_speech_label.position = Vector3(0.0, 0.72, 0.0)
	_speech_label.visible = false
	add_child(_speech_label)

	# Telepathic gaze beam (top_level = true so parent scale/flip never distorts length)
	_telepathy_beam = MeshInstance3D.new()
	_telepathy_beam.top_level = true
	var beam_cyl := CylinderMesh.new()
	beam_cyl.top_radius = 0.011
	beam_cyl.bottom_radius = 0.011
	beam_cyl.height = 1.0
	_telepathy_beam.mesh = beam_cyl
	var beam_mat := StandardMaterial3D.new()
	beam_mat.albedo_color = Color(0.22, 0.56, 0.92, 0.65)
	beam_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	beam_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	_telepathy_beam.material_override = beam_mat
	_telepathy_beam.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_telepathy_beam.visible = false
	add_child(_telepathy_beam)


func _make_ear(pos: Vector3, base_roll: float) -> Node3D:
	var pivot := Node3D.new()
	pivot.position = pos
	pivot.rotation.z = base_roll

	var outer := MeshInstance3D.new()
	var prism := CylinderMesh.new()
	prism.top_radius = 0.004
	prism.bottom_radius = 0.042
	prism.height = 0.098
	outer.mesh = prism
	outer.position = Vector3(0.0, 0.046, 0.0)
	outer.material_override = _body_mat
	pivot.add_child(outer)

	var inner := MeshInstance3D.new()
	var inner_prism := CylinderMesh.new()
	inner_prism.top_radius = 0.002
	inner_prism.bottom_radius = 0.026
	inner_prism.height = 0.072
	inner.mesh = inner_prism
	var pink_mat := StandardMaterial3D.new()
	pink_mat.albedo_color = Color(0.92, 0.66, 0.68)
	pink_mat.roughness = 0.72
	inner.material_override = pink_mat
	inner.position = Vector3(0.0, 0.042, 0.015)
	pivot.add_child(inner)
	return pivot


func _make_eye(pos: Vector3) -> Dictionary:
	var eye_root := Node3D.new()
	eye_root.position = pos
	head_pivot.add_child(eye_root)

	var eye_mi := MeshInstance3D.new()
	var sph := SphereMesh.new()
	sph.radius = 0.025
	sph.height = 0.048
	eye_mi.mesh = sph
	var iris_mat := StandardMaterial3D.new()
	iris_mat.albedo_color = Color(0.86, 0.74, 0.24)
	iris_mat.roughness = 0.22
	eye_mi.material_override = iris_mat
	eye_root.add_child(eye_mi)

	var pupil := MeshInstance3D.new()
	var p_mesh := SphereMesh.new()
	p_mesh.radius = 0.013
	p_mesh.height = 0.026
	pupil.mesh = p_mesh
	var p_mat := StandardMaterial3D.new()
	p_mat.albedo_color = Color(0.05, 0.06, 0.08)
	p_mat.roughness = 0.12
	pupil.material_override = p_mat
	pupil.position = Vector3(0.0, 0.0, 0.016)
	eye_root.add_child(pupil)

	return {"root": eye_root, "pupil": pupil}


func is_gravity_inverted() -> bool:
	return _gravity_inverted


func get_posture_state() -> String:
	return _posture_state


func get_camera_focus_point() -> Vector3:
	if torso_pivot != null and torso_pivot.is_inside_tree():
		return torso_pivot.global_position
	return global_position + Vector3(0.0, TORSO_BASE_HEIGHT, 0.0)


func apply_state(frame: Dictionary, room: RoomBuilder) -> void:
	_room_ref = room

	var morph_var: Variant = frame.get("morphology", {})
	var morph: Dictionary = morph_var if morph_var is Dictionary else {}
	var fur: float = clampf(float(morph.get("fur_density", 0.5)), 0.0, 1.0)
	var mass: float = clampf(float(morph.get("body_mass", 0.5)), 0.0, 1.0)
	var max_anom: float = clampf(float(morph.get("max_anomaly", 0.0)), 0.0, 1.0)

	_body_scale = 0.85 + mass * 0.35
	scale = Vector3(_body_scale, _body_scale, _body_scale)

	var flags_var: Variant = frame.get("anomaly_flags", {})
	var flags: Dictionary = flags_var if flags_var is Dictionary else {}
	_gravity_inverted = bool(flags.get("gravity_inverted", false))
	_phased_active = flags.get("phased_object") != null

	# Target world position: in inverted gravity, seat paws flush against CEILING_Y
	var elev: float = float(frame.get("elevation", 0.0))
	var world_y: float = elev
	if _gravity_inverted:
		world_y = RoomBuilder.CEILING_Y - (TORSO_BASE_HEIGHT * 2.0 * _body_scale)
	_target_pos = RoomBuilder.sim_to_world(
		float(frame.get("x", 3.0)),
		float(frame.get("y", 3.0)),
		world_y,
	)
	if not _has_received_state:
		_has_received_state = true
		position = _target_pos

	# Continuous morphology scaling: hairless skin -> dense fur coat
	_fur_shell_mesh.visible = fur > 0.08
	var fur_scale := 0.96 + fur * 0.16
	_fur_shell_mesh.scale = Vector3(fur_scale, fur_scale, fur_scale)
	_fur_shell_mat.albedo_color.a = clampf(fur * 1.05, 0.0, 0.94)

	var skin_col := Color(0.87, 0.72, 0.65)
	var coat_col := Color(0.36, 0.40, 0.46) if max_anom < 0.8 else Color(0.28, 0.35, 0.50)
	var base_col := skin_col.lerp(coat_col, clampf(fur * 1.25, 0.0, 1.0))
	var accent_col := base_col.lightened(0.14)
	if _phased_active:
		_body_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		_accent_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		base_col.a = 0.36
		accent_col.a = 0.36
	else:
		_body_mat.transparency = BaseMaterial3D.TRANSPARENCY_DISABLED
		_accent_mat.transparency = BaseMaterial3D.TRANSPARENCY_DISABLED
		base_col.a = 1.0
		accent_col.a = 1.0
	_body_mat.albedo_color = base_col
	_accent_mat.albedo_color = accent_col

	# Collar hue subtly distinguishes grounded vs surreal anomalies
	_collar_mat.albedo_color = Color(0.24, 0.48, 0.82) if max_anom >= 0.8 else Color(0.78, 0.30, 0.26)

	# Derive procedural kinematic posture state from live action & PAD emotion
	var action_str: String = str(frame.get("action", "idle"))
	var pad_var: Variant = frame.get("pad", {})
	var pad: Dictionary = pad_var if pad_var is Dictionary else {}
	var arousal: float = float(pad.get("arousal", 0.0))
	var fear: float = float(pad.get("fear_intensity", 0.0))
	_posture_state = _infer_posture(action_str, arousal, fear)

	# Mimicked text or active neuro-symbolic ritual badge
	var mimic: Variant = flags.get("last_mimic_text")
	var active_skill: Variant = frame.get("active_skill_name")
	if mimic != null and not str(mimic).is_empty():
		_speech_label.text = "\"%s\"" % str(mimic)
		_speech_label.visible = true
	elif active_skill != null and not str(active_skill).is_empty():
		_speech_label.text = "[%s]" % str(active_skill)
		_speech_label.visible = true
	else:
		_speech_label.visible = false
	_speech_label.position = Vector3(0.0, -0.22 if _gravity_inverted else 0.72, 0.0)

	# Store telepathic beam target for safe per-frame updates
	var tele_target: Variant = flags.get("telepathic_target")
	if tele_target != null and not str(tele_target).is_empty():
		_telepathy_target_id = str(tele_target)
	else:
		_telepathy_target_id = ""
	_update_telepathy_beam()


func _infer_posture(action_str: String, arousal: float, fear: float) -> String:
	if _gravity_inverted:
		return "ceiling_cling"
	if _last_speed > 0.14 or "approach:" in action_str:
		return "walking"
	if (
		"hide_and_nap" in action_str
		or "sunbathe" in action_str
		or "curl_for_warmth" in action_str
		or "sit_on_keyboard" in action_str
		or (action_str == "idle" and arousal < -0.15)
	):
		return "sleeping_curled"
	if (
		"crouch" in action_str
		or "knock_over" in action_str
		or "nudge" in action_str
		or "phase_object" in action_str
		or fear > 0.42
	):
		return "crouched"
	return "sitting_alert"


func _update_telepathy_beam() -> void:
	if _telepathy_beam == null:
		return
	if (
		_telepathy_target_id.is_empty()
		or _room_ref == null
		or not is_inside_tree()
		or not _telepathy_beam.is_inside_tree()
		or head_pivot == null
		or not head_pivot.is_inside_tree()
	):
		_telepathy_beam.visible = false
		return

	var dst := _room_ref.get_object_focus_point(_telepathy_target_id)
	var src := head_pivot.global_position
	var diff := dst - src
	var dist := diff.length()
	if dist <= 0.06:
		_telepathy_beam.visible = false
		return

	var dir := diff / dist
	var up_vec := Vector3.RIGHT if absf(dir.dot(Vector3.UP)) > 0.99 else Vector3.UP
	_telepathy_beam.visible = true
	_telepathy_beam.global_position = src + diff * 0.5
	_telepathy_beam.look_at(dst, up_vec)
	_telepathy_beam.rotate_object_local(Vector3.RIGHT, PI * 0.5)
	_telepathy_beam.scale = Vector3(1.0, dist, 1.0)


func tick_motion(delta: float) -> float:
	var prev := position
	var pos_alpha := 1.0 - exp(-7.5 * delta)
	position = position.lerp(_target_pos, pos_alpha)

	var vel := (position - prev) / maxf(delta, 0.001)
	var horiz_vel := Vector2(vel.x, vel.z)
	_last_speed = horiz_vel.length()

	if _last_speed > 0.08:
		var target_yaw := atan2(vel.x, vel.z)
		var yaw_alpha := 1.0 - exp(-9.0 * delta)
		rotation.y = lerp_angle(rotation.y, target_yaw, yaw_alpha)

	# Smoothly flip around torso center (flip_pivot) instead of ground pivot
	var target_roll := PI if _gravity_inverted else 0.0
	var roll_alpha := 1.0 - exp(-7.0 * delta)
	flip_pivot.rotation.z = lerp_angle(flip_pivot.rotation.z, target_roll, roll_alpha)

	# Update contact shadow placement & opacity against floor or ceiling
	if _shadow_disc != null and _shadow_mat != null:
		if _gravity_inverted:
			var ceil_local_y := (RoomBuilder.CEILING_Y - 0.008 - position.y) / maxf(_body_scale, 0.1)
			_shadow_disc.position = Vector3(0.0, ceil_local_y, 0.0)
			_shadow_mat.albedo_color.a = 0.22
		else:
			_shadow_disc.position = Vector3(0.0, 0.006, 0.0)
			_shadow_mat.albedo_color.a = 0.24

	_update_telepathy_beam()
	return _last_speed
