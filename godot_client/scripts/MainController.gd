extends Node3D

var _bridge: BrainBridge
var _room: RoomBuilder
var _pet: ProceduralPet
var _expr: MicroExpressionController
var _camera_rig: OrbitCamera
var _hud: TelemetryHUD

var _latest_frame: Dictionary = {}
var _floor_plane: Plane = Plane(Vector3.UP, 0.0)
var _lmb_down_pos: Vector2 = Vector2.ZERO
var _lmb_tracking: bool = false


func _ready() -> void:
	_setup_studio_environment()

	_room = RoomBuilder.new()
	add_child(_room)
	_room.build_room()

	_pet = ProceduralPet.new()
	add_child(_pet)
	_pet.build_rig()

	_expr = MicroExpressionController.new()
	add_child(_expr)

	_camera_rig = OrbitCamera.new()
	add_child(_camera_rig)
	_camera_rig.set_follow_target(_pet)
	_camera_rig.follow_changed.connect(_on_follow_changed)

	_hud = TelemetryHUD.new()
	add_child(_hud)
	_hud.build_ui()
	_hud.command_requested.connect(_on_hud_command)
	_hud.follow_toggled.connect(func(enabled: bool) -> void: _camera_rig.set_follow(enabled))
	_hud.camera_preset_requested.connect(func(pid: int) -> void: _camera_rig.apply_preset(pid))
	_hud.archive_refresh_requested.connect(func() -> void: _bridge.fetch_archive())

	_bridge = BrainBridge.new()
	add_child(_bridge)
	_bridge.state_received.connect(_on_state_received)
	_bridge.connection_changed.connect(_on_connection_changed)
	_bridge.archive_loaded.connect(_on_archive_loaded)
	_bridge.archive_error.connect(_on_archive_error)


func _setup_studio_environment() -> void:
	var world_env := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.945, 0.955, 0.972)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.96, 0.97, 0.99)
	env.ambient_light_energy = 0.62
	world_env.environment = env
	add_child(world_env)

	# Primary warm-neutral studio key light with crisp shadows
	var sun := DirectionalLight3D.new()
	sun.light_color = Color(1.0, 0.985, 0.96)
	sun.light_energy = 1.08
	sun.shadow_enabled = true
	sun.shadow_bias = 0.035
	sun.shadow_blur = 1.4
	sun.rotation_degrees = Vector3(-50.0, 34.0, 0.0)
	add_child(sun)

	# Subtle shadowless cool studio fill light to soften dark undersides
	var fill := DirectionalLight3D.new()
	fill.light_color = Color(0.84, 0.91, 1.0)
	fill.light_energy = 0.30
	fill.shadow_enabled = false
	fill.rotation_degrees = Vector3(-28.0, -142.0, 0.0)
	add_child(fill)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey:
		var key_ev: InputEventKey = event
		if key_ev.pressed and not key_ev.echo:
			if key_ev.keycode == KEY_ESCAPE and _hud.is_text_input_focused():
				var owner := get_viewport().gui_get_focus_owner()
				if owner != null:
					owner.release_focus()
				return
			# Never fire simulation/camera shortcuts while typing in any UI input control
			if _hud.is_text_input_focused():
				return
			_handle_hotkey(key_ev.keycode)
		return

	if event is InputEventMouseButton:
		var mb: InputEventMouseButton = event
		if mb.button_index == MOUSE_BUTTON_LEFT:
			if mb.pressed:
				_lmb_down_pos = mb.position
				_lmb_tracking = true
			elif _lmb_tracking:
				_lmb_tracking = false
				if mb.position.distance_to(_lmb_down_pos) < 8.0:
					_try_move_user_at_screen_pos(mb.position)


func _handle_hotkey(keycode: int) -> void:
	match keycode:
		KEY_SPACE:
			var currently_paused: bool = bool(_latest_frame.get("paused", false))
			_bridge.send_command({"type": "set_paused", "paused": not currently_paused})
		KEY_F:
			_camera_rig.toggle_follow()
		KEY_1, KEY_R:
			_camera_rig.apply_preset(1)
		KEY_2:
			_camera_rig.apply_preset(2)
		KEY_3:
			_camera_rig.apply_preset(3)
		KEY_TAB:
			_hud.toggle_side_panel()


func _process(delta: float) -> void:
	var speed := _pet.tick_motion(delta)
	if not _latest_frame.is_empty():
		var target_var: Variant = _latest_frame.get("target_object_id", "user")
		var target_id: String = str(target_var) if target_var != null else "user"
		var focus := _room.get_object_focus_point(target_id)
		var pad_var: Variant = _latest_frame.get("pad", {})
		var pad: Dictionary = pad_var if pad_var is Dictionary else {}
		_expr.update_expressions(_pet, pad, focus, speed, delta)

	_update_hover_inspection()


func _try_move_user_at_screen_pos(screen_pos: Vector2) -> void:
	var cam := _camera_rig.get_camera()
	if cam == null:
		return
	var origin := cam.project_ray_origin(screen_pos)
	var dir := cam.project_ray_normal(screen_pos)
	var hit: Variant = _floor_plane.intersects_ray(origin, dir)
	if not (hit is Vector3):
		return
	var hit_pos: Vector3 = hit
	if absf(hit_pos.x) > 3.05 or absf(hit_pos.z) > 3.05:
		return
	var sim_xy := RoomBuilder.world_to_sim(hit_pos)
	_room.spawn_floor_ripple(Vector3(clampf(hit_pos.x, -3.0, 3.0), 0.0, clampf(hit_pos.z, -3.0, 3.0)))
	_bridge.send_command({
		"type": "move_user",
		"x": snappedf(sim_xy.x, 0.01),
		"y": snappedf(sim_xy.y, 0.01),
	})


func _update_hover_inspection() -> void:
	if get_viewport().gui_get_hovered_control() != null:
		_room.set_hovered_object("")
		_hud.update_hover_tooltip({}, Vector2.ZERO)
		return

	var cam := _camera_rig.get_camera()
	if cam == null:
		return
	var mouse_pos := get_viewport().get_mouse_position()
	var origin := cam.project_ray_origin(mouse_pos)
	var dir := cam.project_ray_normal(mouse_pos)
	var hit_id := _room.pick_object_from_ray(origin, dir)
	_room.set_hovered_object(hit_id)
	if hit_id.is_empty():
		_hud.update_hover_tooltip({}, mouse_pos)
	else:
		_hud.update_hover_tooltip(_room.get_object_meta(hit_id), mouse_pos)


func _on_hud_command(payload: Dictionary) -> void:
	_bridge.send_command(payload)


func _on_follow_changed(enabled: bool) -> void:
	_hud.set_follow_active(enabled)


func _on_connection_changed(is_connected: bool) -> void:
	_hud.set_connection_status(is_connected)


func _on_archive_loaded(elites: Array, metadata: Dictionary) -> void:
	_hud.set_archive_elites(elites, metadata)


func _on_archive_error(message: String) -> void:
	_hud.set_archive_status(message)


func _on_state_received(state: Dictionary) -> void:
	_latest_frame = state
	var objs_var: Variant = state.get("world_objects", [])
	var objs: Array = objs_var if objs_var is Array else []
	_room.sync_objects(objs)
	_pet.apply_state(state, _room)
	_hud.update_frame(state)
