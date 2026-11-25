class_name OrbitCamera
extends Node3D

signal follow_changed(enabled: bool)

const MIN_DISTANCE: float = 1.35
const MAX_DISTANCE: float = 12.5
const MIN_PITCH_RAD: float = -0.28  # ~-16 deg (lets user tilt up to watch ceiling walk)
const MAX_PITCH_RAD: float = 1.47   # ~84 deg (top-down without gimbal lock)
const MIN_CAMERA_Y: float = 0.18

var follow_enabled: bool = false

var _camera: Camera3D = null
var _follow_pet: ProceduralPet = null

var _target: Vector3 = Vector3(0.0, 0.55, 0.0)
var _desired_target: Vector3 = Vector3(0.0, 0.55, 0.0)
var _yaw: float = deg_to_rad(42.0)
var _desired_yaw: float = deg_to_rad(42.0)
var _pitch: float = deg_to_rad(35.0)
var _desired_pitch: float = deg_to_rad(35.0)
var _distance: float = 7.6
var _desired_distance: float = 7.6

var _orbiting: bool = false
var _panning: bool = false


func _ready() -> void:
	_camera = Camera3D.new()
	_camera.name = "StudioCamera"
	_camera.fov = 44.0
	_camera.near = 0.05
	_camera.far = 80.0
	add_child(_camera)
	_camera.make_current()
	_apply_camera_transform(1.0)


func get_camera() -> Camera3D:
	return _camera


func set_follow_target(pet: ProceduralPet) -> void:
	_follow_pet = pet


func set_follow(enabled: bool) -> void:
	if follow_enabled == enabled:
		return
	follow_enabled = enabled
	follow_changed.emit(follow_enabled)


func toggle_follow() -> void:
	set_follow(not follow_enabled)


func apply_preset(preset_id: int) -> void:
	match preset_id:
		2:
			# Close-up Pet Cam (auto-enables smooth pet follow)
			set_follow(true)
			_desired_pitch = deg_to_rad(18.0)
			_desired_distance = 2.15
		3:
			# Top-Down Architectural CAD View
			set_follow(false)
			_desired_target = Vector3(0.0, 0.20, 0.0)
			_desired_yaw = 0.0
			_desired_pitch = deg_to_rad(83.5)
			_desired_distance = 8.4
		_:
			# Preset 1 / Reset: Studio 3/4 Overview
			set_follow(false)
			_desired_target = Vector3(0.0, 0.55, 0.0)
			_desired_yaw = deg_to_rad(42.0)
			_desired_pitch = deg_to_rad(35.0)
			_desired_distance = 7.6


func _is_text_input_focused() -> bool:
	var owner := get_viewport().gui_get_focus_owner()
	if owner == null:
		return false
	return owner is LineEdit or owner is TextEdit or owner is CodeEdit or owner is SpinBox


func _input(event: InputEvent) -> void:
	# Always capture drag release & active drag motion even if cursor moves over HUD panels
	if event is InputEventMouseButton:
		var mb: InputEventMouseButton = event
		if not mb.pressed:
			if mb.button_index == MOUSE_BUTTON_RIGHT:
				_orbiting = false
			elif mb.button_index == MOUSE_BUTTON_MIDDLE:
				_panning = false
	elif event is InputEventMouseMotion and (_orbiting or _panning):
		var mm: InputEventMouseMotion = event
		if _orbiting:
			_desired_yaw -= mm.relative.x * 0.0068
			_desired_pitch = clampf(
				_desired_pitch + mm.relative.y * 0.0058,
				MIN_PITCH_RAD,
				MAX_PITCH_RAD,
			)
		elif _panning:
			set_follow(false)
			var pan_scale := _distance * 0.0016
			var flat_right := Vector3(cos(_yaw), 0.0, -sin(_yaw)).normalized()
			var flat_fwd := Vector3(-sin(_yaw), 0.0, -cos(_yaw)).normalized()
			_desired_target -= flat_right * (mm.relative.x * pan_scale)
			_desired_target += flat_fwd * (mm.relative.y * pan_scale)
			_clamp_desired_target()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb: InputEventMouseButton = event
		if mb.pressed and mb.button_index == MOUSE_BUTTON_RIGHT:
			_orbiting = true
		elif mb.pressed and mb.button_index == MOUSE_BUTTON_MIDDLE:
			_panning = true
		elif mb.pressed and mb.button_index == MOUSE_BUTTON_WHEEL_UP:
			_desired_distance = clampf(_desired_distance * 0.89, MIN_DISTANCE, MAX_DISTANCE)
		elif mb.pressed and mb.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_desired_distance = clampf(_desired_distance * 1.12, MIN_DISTANCE, MAX_DISTANCE)


func _process(delta: float) -> void:
	if not _is_text_input_focused():
		_handle_keyboard_pan(delta)

	if follow_enabled and _follow_pet != null:
		var focus := _follow_pet.get_camera_focus_point()
		_desired_target = focus

	var damp := 1.0 - exp(-10.5 * delta)
	_apply_camera_transform(damp)


func _handle_keyboard_pan(delta: float) -> void:
	var move_x := 0.0
	var move_z := 0.0
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT):
		move_x += 1.0
	if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT):
		move_x -= 1.0
	if Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP):
		move_z += 1.0
	if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN):
		move_z -= 1.0

	if absf(move_x) > 0.01 or absf(move_z) > 0.01:
		set_follow(false)
		var speed := clampf(_distance * 0.52, 2.0, 6.0)
		var flat_right := Vector3(cos(_yaw), 0.0, -sin(_yaw)).normalized()
		var flat_fwd := Vector3(-sin(_yaw), 0.0, -cos(_yaw)).normalized()
		var dir := (flat_right * move_x + flat_fwd * move_z).normalized()
		_desired_target += dir * speed * delta
		_clamp_desired_target()


func _clamp_desired_target() -> void:
	_desired_target.x = clampf(_desired_target.x, -3.4, 3.4)
	_desired_target.y = clampf(_desired_target.y, 0.12, 2.35)
	_desired_target.z = clampf(_desired_target.z, -3.4, 3.4)


func _apply_camera_transform(alpha: float) -> void:
	if _camera == null or not _camera.is_inside_tree():
		return
	_target = _target.lerp(_desired_target, alpha)
	_yaw = lerp_angle(_yaw, _desired_yaw, alpha)
	_pitch = lerpf(_pitch, _desired_pitch, alpha)
	_distance = lerpf(_distance, _desired_distance, alpha)

	var horiz := _distance * cos(_pitch)
	var offset := Vector3(
		horiz * sin(_yaw),
		_distance * sin(_pitch),
		horiz * cos(_yaw),
	)
	var cam_pos := _target + offset
	cam_pos.y = maxf(cam_pos.y, MIN_CAMERA_Y)
	_camera.global_position = cam_pos

	var to_target := _target - cam_pos
	if to_target.length() > 0.04:
		var look_dir := to_target.normalized()
		var up_vec := Vector3.FORWARD if absf(look_dir.dot(Vector3.UP)) > 0.995 else Vector3.UP
		_camera.look_at(_target, up_vec)
