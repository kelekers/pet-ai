class_name RoomBuilder
extends Node3D

const ROOM_SIZE_M: float = 6.0
const CEILING_Y: float = 2.40

var object_nodes: Dictionary = {}
var object_meta: Dictionary = {}

var _target_positions: Dictionary = {}
var _object_bounds: Dictionary = {}  # obj_id -> Dictionary {center_offset: Vector3, radius: float}

var _glass_mesh: MeshInstance3D = null
var _water_fill_mesh: MeshInstance3D = null
var _puddle_mesh: MeshInstance3D = null
var _radiator_glow_mat: StandardMaterial3D = null
var _laptop_screen_mat: StandardMaterial3D = null
var _sunbeam_disc_mat: StandardMaterial3D = null
var _sunbeam_motes: Array[MeshInstance3D] = []
var _sunbeam_base_offsets: Array[Vector3] = []

var _hover_ring: MeshInstance3D = null
var _hover_ring_mat: StandardMaterial3D = null
var _hovered_id: String = ""

var _ripples: Array[Dictionary] = []
var _time: float = 0.0


static func sim_to_world(sim_x: float, sim_y: float, elevation: float = 0.0) -> Vector3:
	return Vector3(sim_x - 3.0, elevation, sim_y - 3.0)


static func world_to_sim(world_pos: Vector3) -> Vector2:
	return Vector2(
		clampf(world_pos.x + 3.0, 0.0, ROOM_SIZE_M),
		clampf(world_pos.z + 3.0, 0.0, ROOM_SIZE_M),
	)


func build_room() -> void:
	# Clean matte light-mode studio floor (6m x 6m)
	var floor_mesh := MeshInstance3D.new()
	var plane := PlaneMesh.new()
	plane.size = Vector2(6.5, 6.5)
	floor_mesh.mesh = plane
	var floor_mat := StandardMaterial3D.new()
	floor_mat.albedo_color = Color(0.92, 0.93, 0.945)
	floor_mat.roughness = 0.88
	floor_mesh.material_override = floor_mat
	add_child(floor_mesh)

	# Subtle recessed room slab trim
	var slab := _add_box(
		self,
		Vector3(6.56, 0.08, 6.56),
		Vector3(0.0, -0.042, 0.0),
		Color(0.84, 0.86, 0.89),
		0.9,
	)
	slab.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF

	# Architectural 1m CAD grid lines on the floor
	var major_grid_mat := StandardMaterial3D.new()
	major_grid_mat.albedo_color = Color(0.76, 0.80, 0.84)
	major_grid_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED

	var minor_grid_mat := StandardMaterial3D.new()
	minor_grid_mat.albedo_color = Color(0.85, 0.87, 0.90)
	minor_grid_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED

	for step in range(-6, 7):
		var coord := float(step) * 0.5
		var is_major := (step % 2) == 0
		var mat := major_grid_mat if is_major else minor_grid_mat
		var thickness := 0.014 if is_major else 0.006

		var line_x := MeshInstance3D.new()
		var box_x := BoxMesh.new()
		box_x.size = Vector3(6.0, 0.003, thickness)
		line_x.mesh = box_x
		line_x.material_override = mat
		line_x.position = Vector3(0.0, 0.002, coord)
		line_x.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		add_child(line_x)

		var line_z := MeshInstance3D.new()
		var box_z := BoxMesh.new()
		box_z.size = Vector3(thickness, 0.003, 6.0)
		line_z.mesh = box_z
		line_z.material_override = mat
		line_z.position = Vector3(coord, 0.002, 0.0)
		line_z.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		add_child(line_z)

	# Perimeter architectural walls (back + left for unobstructed studio camera)
	_add_wall(Vector3(0.0, CEILING_Y * 0.5, -3.12), Vector3(6.48, CEILING_Y, 0.14), Color(0.955, 0.962, 0.975))
	_add_wall(Vector3(-3.12, CEILING_Y * 0.5, 0.0), Vector3(0.14, CEILING_Y, 6.48), Color(0.935, 0.945, 0.962))
	# Baseboard molding trim
	_add_box(self, Vector3(6.24, 0.09, 0.03), Vector3(0.0, 0.045, -3.04), Color(0.84, 0.86, 0.89), 0.85)
	_add_box(self, Vector3(0.03, 0.09, 6.24), Vector3(-3.04, 0.045, 0.0), Color(0.82, 0.84, 0.88), 0.85)

	# Faint architectural ceiling grid plane at CEILING_Y (visible when looking up for invert_gravity)
	_build_ceiling_grid()

	# Hover inspection highlight ring
	_hover_ring = MeshInstance3D.new()
	var ring_mesh := TorusMesh.new()
	ring_mesh.inner_radius = 0.38
	ring_mesh.outer_radius = 0.43
	ring_mesh.rings = 24
	ring_mesh.ring_segments = 12
	_hover_ring.mesh = ring_mesh
	_hover_ring_mat = StandardMaterial3D.new()
	_hover_ring_mat.albedo_color = Color(0.18, 0.52, 0.88, 0.72)
	_hover_ring_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	_hover_ring_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	_hover_ring.material_override = _hover_ring_mat
	_hover_ring.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_hover_ring.visible = false
	add_child(_hover_ring)


func _build_ceiling_grid() -> void:
	var ceil_root := Node3D.new()
	ceil_root.name = "CeilingGrid"
	ceil_root.position = Vector3(0.0, CEILING_Y, 0.0)
	add_child(ceil_root)

	# Translucent ceiling glass plane (only visible from underside, never casts shadow)
	var ceil_plane := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(6.0, 6.0)
	ceil_plane.mesh = pm
	ceil_plane.rotation.x = PI  # Face normal downward (-Y) so top-down camera sees straight through!
	var ceil_mat := StandardMaterial3D.new()
	ceil_mat.albedo_color = Color(0.86, 0.90, 0.96, 0.14)
	ceil_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	ceil_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	ceil_plane.material_override = ceil_mat
	ceil_plane.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	ceil_root.add_child(ceil_plane)

	# Crisp low-opacity CAD ceiling grid lines (shadowless)
	var cgrid_mat := StandardMaterial3D.new()
	cgrid_mat.albedo_color = Color(0.55, 0.64, 0.76, 0.24)
	cgrid_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	cgrid_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	for i in range(-3, 4):
		var lx := MeshInstance3D.new()
		var bx := BoxMesh.new()
		bx.size = Vector3(6.0, 0.004, 0.012)
		lx.mesh = bx
		lx.material_override = cgrid_mat
		lx.position = Vector3(0.0, 0.0, float(i))
		lx.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		ceil_root.add_child(lx)

		var lz := MeshInstance3D.new()
		var bz := BoxMesh.new()
		bz.size = Vector3(0.012, 0.004, 6.0)
		lz.mesh = bz
		lz.material_override = cgrid_mat
		lz.position = Vector3(float(i), 0.0, 0.0)
		lz.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		ceil_root.add_child(lz)


func _process(delta: float) -> void:
	_time += delta

	# Frame-rate-independent 60 FPS exponential damping for all Smart Objects
	var alpha := 1.0 - exp(-10.0 * delta)
	for obj_id in _target_positions.keys():
		if not object_nodes.has(obj_id):
			continue
		var node: Node3D = object_nodes[obj_id]
		var target: Vector3 = _target_positions[obj_id]
		node.position = node.position.lerp(target, alpha)

	# Animate subtle sunbeam shimmer and floating warmth motes
	if _sunbeam_disc_mat != null:
		var pulse := 0.44 + 0.08 * sin(_time * 1.9)
		_sunbeam_disc_mat.albedo_color.a = pulse
	for i in range(_sunbeam_motes.size()):
		var mote: MeshInstance3D = _sunbeam_motes[i]
		var base: Vector3 = _sunbeam_base_offsets[i]
		var drift_y := fmod(_time * 0.14 + float(i) * 0.23, 0.85)
		mote.position = Vector3(
			base.x + sin(_time * 1.2 + float(i) * 1.7) * 0.05,
			0.06 + drift_y,
			base.z + cos(_time * 1.1 + float(i) * 1.3) * 0.05,
		)

	# Animate subtle radiator thermal pulse
	if _radiator_glow_mat != null:
		_radiator_glow_mat.emission_energy_multiplier = 0.55 + 0.15 * sin(_time * 2.4)

	# Update hover ring transform + gentle pulse
	if _hover_ring != null and _hover_ring.visible and object_nodes.has(_hovered_id):
		var hovered_node: Node3D = object_nodes[_hovered_id]
		_hover_ring.global_position = hovered_node.global_position + Vector3(0.0, 0.02, 0.0)
		var ring_pulse := 1.0 + 0.04 * sin(_time * 5.0)
		_hover_ring.scale = Vector3(ring_pulse, 1.0, ring_pulse)

	# Update expanding floor click ripples
	var idx := _ripples.size() - 1
	while idx >= 0:
		var item: Dictionary = _ripples[idx]
		var age: float = float(item.get("age", 0.0)) + delta
		var dur: float = float(item.get("duration", 0.55))
		var mesh_node: MeshInstance3D = item["node"]
		var mat: StandardMaterial3D = item["mat"]
		if age >= dur:
			if is_instance_valid(mesh_node):
				mesh_node.queue_free()
			_ripples.remove_at(idx)
		else:
			item["age"] = age
			var t := age / dur
			var ease_out := 1.0 - pow(1.0 - t, 3.0)
			var s := lerpf(0.25, 1.55, ease_out)
			mesh_node.scale = Vector3(s, 1.0, s)
			mat.albedo_color.a = (1.0 - t) * 0.78
		idx -= 1


func spawn_floor_ripple(world_pos: Vector3) -> void:
	var ring := MeshInstance3D.new()
	var torus := TorusMesh.new()
	torus.inner_radius = 0.24
	torus.outer_radius = 0.28
	torus.rings = 24
	torus.ring_segments = 10
	ring.mesh = torus
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.16, 0.48, 0.85, 0.78)
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	ring.material_override = mat
	ring.position = Vector3(world_pos.x, 0.015, world_pos.z)
	ring.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(ring)
	_ripples.append({"node": ring, "mat": mat, "age": 0.0, "duration": 0.55})


func sync_objects(world_objects: Array) -> void:
	for raw in world_objects:
		if not (raw is Dictionary):
			continue
		var data: Dictionary = raw
		var obj_id: String = str(data.get("obj_id", ""))
		if obj_id.is_empty():
			continue

		object_meta[obj_id] = data
		var target_pos := sim_to_world(
			float(data.get("x", 3.0)),
			float(data.get("y", 3.0)),
			0.0,
		)

		if not object_nodes.has(obj_id):
			var node := _create_smart_object(data)
			node.position = target_pos
			object_nodes[obj_id] = node
			add_child(node)

		_target_positions[obj_id] = target_pos

		if obj_id == "water_glass" and _glass_mesh != null and _puddle_mesh != null:
			var intact: bool = bool(data.get("is_intact", true))
			_glass_mesh.visible = intact
			if _water_fill_mesh != null:
				_water_fill_mesh.visible = intact
			_puddle_mesh.visible = not intact
		elif obj_id == "sunbeam" and _sunbeam_disc_mat != null:
			var temp_c: float = float(data.get("surface_temp_c", 29.0))
			var warmth_t := clampf((temp_c - 24.0) / 10.0, 0.0, 1.0)
			var col := Color(0.98, 0.92, 0.68, _sunbeam_disc_mat.albedo_color.a).lerp(
				Color(1.0, 0.88, 0.44, _sunbeam_disc_mat.albedo_color.a),
				warmth_t,
			)
			_sunbeam_disc_mat.albedo_color = col


func get_object_focus_point(obj_id: String) -> Vector3:
	if object_nodes.has(obj_id):
		var n: Node3D = object_nodes[obj_id]
		var elev: float = 0.0
		if object_meta.has(obj_id):
			var meta: Dictionary = object_meta[obj_id]
			elev = float(meta.get("elevation", 0.0))
			if bool(meta.get("is_user", false)):
				return n.global_position + Vector3(0.0, 1.15, 0.0)
		return n.global_position + Vector3(0.0, maxf(0.32, elev + 0.25), 0.0)
	return Vector3(0.0, 0.45, 0.0)


func pick_object_from_ray(ray_origin: Vector3, ray_dir: Vector3) -> String:
	var dir := ray_dir.normalized()
	var best_id := ""
	var best_t := 9999.0

	for obj_id_var in object_nodes.keys():
		var obj_id: String = str(obj_id_var)
		var node: Node3D = object_nodes[obj_id]
		var bounds: Dictionary = _object_bounds.get(
			obj_id,
			{"offset": Vector3(0.0, 0.35, 0.0), "radius": 0.42},
		)
		var center: Vector3 = node.global_position + (bounds["offset"] as Vector3)
		var radius: float = float(bounds["radius"])

		var oc := ray_origin - center
		var b := oc.dot(dir)
		var c := oc.dot(oc) - radius * radius
		var disc := b * b - c
		if disc < 0.0:
			continue
		var t := -b - sqrt(disc)
		if t > 0.05 and t < best_t:
			best_t = t
			best_id = obj_id
	return best_id


func set_hovered_object(obj_id: String) -> void:
	if _hovered_id == obj_id:
		return
	_hovered_id = obj_id
	if _hover_ring == null:
		return
	if obj_id.is_empty() or not object_nodes.has(obj_id):
		_hover_ring.visible = false
		return
	var node: Node3D = object_nodes[obj_id]
	var bounds: Dictionary = _object_bounds.get(obj_id, {"radius": 0.45})
	var r: float = clampf(float(bounds.get("radius", 0.45)), 0.25, 0.95)
	_hover_ring.global_position = node.global_position + Vector3(0.0, 0.02, 0.0)
	var torus := _hover_ring.mesh as TorusMesh
	if torus != null:
		torus.inner_radius = r * 0.88
		torus.outer_radius = r * 0.98
	_hover_ring.visible = true


func get_object_meta(obj_id: String) -> Dictionary:
	if object_meta.has(obj_id):
		return object_meta[obj_id] as Dictionary
	return {}


func _add_wall(pos: Vector3, sz: Vector3, col: Color) -> void:
	var wall := MeshInstance3D.new()
	var box := BoxMesh.new()
	box.size = sz
	wall.mesh = box
	var mat := StandardMaterial3D.new()
	mat.albedo_color = col
	mat.roughness = 0.92
	wall.material_override = mat
	wall.position = pos
	add_child(wall)


func _create_smart_object(data: Dictionary) -> Node3D:
	var obj_id: String = str(data.get("obj_id", "obj"))
	var root := Node3D.new()
	root.name = obj_id
	root.position = sim_to_world(float(data.get("x", 3.0)), float(data.get("y", 3.0)), 0.0)

	var elev: float = float(data.get("elevation", 0.0))

	match obj_id:
		"radiator":
			# Architectural finned radiator with subtle warm emissive core
			_add_box(root, Vector3(0.18, 0.62, 1.06), Vector3(0.0, 0.34, 0.0), Color(0.86, 0.87, 0.89), 0.55)
			_radiator_glow_mat = StandardMaterial3D.new()
			_radiator_glow_mat.albedo_color = Color(0.92, 0.46, 0.32)
			_radiator_glow_mat.emission_enabled = true
			_radiator_glow_mat.emission = Color(1.0, 0.48, 0.22)
			_radiator_glow_mat.emission_energy_multiplier = 0.65
			_radiator_glow_mat.roughness = 0.55
			for i in range(6):
				var z_off := -0.40 + float(i) * 0.16
				var fin := MeshInstance3D.new()
				var fin_box := BoxMesh.new()
				fin_box.size = Vector3(0.24, 0.54, 0.06)
				fin.mesh = fin_box
				fin.position = Vector3(0.02, 0.34, z_off)
				fin.material_override = _radiator_glow_mat
				root.add_child(fin)
			_object_bounds[obj_id] = {"offset": Vector3(0.0, 0.35, 0.0), "radius": 0.62}

		"laptop":
			# Studio desk with legs + warm laptop with glowing screen
			_add_box(root, Vector3(1.18, 0.055, 0.68), Vector3(0.16, elev - 0.028, 0.0), Color(0.84, 0.77, 0.67), 0.72)
			for lx in [-0.38, 0.70]:
				for lz in [-0.28, 0.28]:
					_add_box(root, Vector3(0.05, elev - 0.055, 0.05), Vector3(lx, (elev - 0.055) * 0.5, lz), Color(0.32, 0.35, 0.40), 0.6)
			# Laptop chassis + keyboard deck
			_add_box(root, Vector3(0.38, 0.025, 0.26), Vector3(0.0, elev + 0.014, 0.0), Color(0.26, 0.30, 0.36), 0.45)
			_add_box(root, Vector3(0.32, 0.008, 0.14), Vector3(0.0, elev + 0.028, -0.01), Color(0.18, 0.20, 0.24), 0.7)
			# Laptop lid + glowing display
			_add_box(root, Vector3(0.38, 0.25, 0.018), Vector3(0.0, elev + 0.14, -0.13), Color(0.22, 0.25, 0.31), 0.42)
			var screen := MeshInstance3D.new()
			var scr_box := BoxMesh.new()
			scr_box.size = Vector3(0.34, 0.21, 0.006)
			screen.mesh = scr_box
			_laptop_screen_mat = StandardMaterial3D.new()
			_laptop_screen_mat.albedo_color = Color(0.76, 0.89, 0.99)
			_laptop_screen_mat.emission_enabled = true
			_laptop_screen_mat.emission = Color(0.56, 0.78, 0.98)
			_laptop_screen_mat.emission_energy_multiplier = 0.85
			screen.material_override = _laptop_screen_mat
			screen.position = Vector3(0.0, elev + 0.14, -0.119)
			root.add_child(screen)
			_object_bounds[obj_id] = {"offset": Vector3(0.08, elev, 0.0), "radius": 0.68}

		"sunbeam":
			# Soft multi-layer warm sunbeam pool + floating warmth motes
			var outer := MeshInstance3D.new()
			var outer_cyl := CylinderMesh.new()
			outer_cyl.top_radius = 0.96
			outer_cyl.bottom_radius = 0.96
			outer_cyl.height = 0.008
			outer.mesh = outer_cyl
			var outer_mat := StandardMaterial3D.new()
			outer_mat.albedo_color = Color(1.0, 0.94, 0.68, 0.22)
			outer_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
			outer_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
			outer.material_override = outer_mat
			outer.position = Vector3(0.0, 0.006, 0.0)
			outer.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			root.add_child(outer)

			var disc := MeshInstance3D.new()
			var cyl := CylinderMesh.new()
			cyl.top_radius = 0.72
			cyl.bottom_radius = 0.72
			cyl.height = 0.012
			disc.mesh = cyl
			_sunbeam_disc_mat = StandardMaterial3D.new()
			_sunbeam_disc_mat.albedo_color = Color(1.0, 0.90, 0.50, 0.48)
			_sunbeam_disc_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
			_sunbeam_disc_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
			disc.material_override = _sunbeam_disc_mat
			disc.position = Vector3(0.0, 0.009, 0.0)
			disc.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			root.add_child(disc)

			var mote_mat := StandardMaterial3D.new()
			mote_mat.albedo_color = Color(1.0, 0.95, 0.72, 0.62)
			mote_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
			mote_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
			for m in range(7):
				var angle := float(m) * (TAU / 7.0)
				var rad := 0.22 + fmod(float(m) * 0.17, 0.42)
				var base_off := Vector3(cos(angle) * rad, 0.1, sin(angle) * rad)
				var mote := MeshInstance3D.new()
				var sph := SphereMesh.new()
				sph.radius = 0.016
				sph.height = 0.032
				mote.mesh = sph
				mote.material_override = mote_mat
				mote.position = base_off
				mote.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
				root.add_child(mote)
				_sunbeam_motes.append(mote)
				_sunbeam_base_offsets.append(base_off)
			_object_bounds[obj_id] = {"offset": Vector3(0.0, 0.15, 0.0), "radius": 0.85}

		"water_glass":
			_glass_mesh = _add_cylinder(
				root, 0.056, 0.145, Vector3(0.0, elev + 0.072, 0.0), Color(0.76, 0.91, 0.98, 0.55), true
			)
			_water_fill_mesh = _add_cylinder(
				root, 0.048, 0.10, Vector3(0.0, elev + 0.055, 0.0), Color(0.28, 0.68, 0.92, 0.78), true
			)
			_puddle_mesh = _add_cylinder(
				root, 0.30, 0.009, Vector3(0.10, 0.008, 0.12), Color(0.32, 0.72, 0.94, 0.56), true
			)
			_puddle_mesh.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			_puddle_mesh.visible = false
			_object_bounds[obj_id] = {"offset": Vector3(0.0, elev + 0.08, 0.0), "radius": 0.24}

		"food_bowl":
			_add_cylinder(root, 0.19, 0.075, Vector3(0.0, 0.038, 0.0), Color(0.24, 0.56, 0.52))
			_add_cylinder(root, 0.15, 0.03, Vector3(0.0, 0.068, 0.0), Color(0.56, 0.42, 0.30))
			_object_bounds[obj_id] = {"offset": Vector3(0.0, 0.08, 0.0), "radius": 0.28}

		"cardboard_box":
			# Open-top architectural cardboard box so the cat visibly curls inside
			var box_col := Color(0.77, 0.64, 0.46)
			var wall_t := 0.028
			_add_box(root, Vector3(0.62, 0.025, 0.62), Vector3(0.0, 0.013, 0.0), Color(0.68, 0.55, 0.38), 0.9)
			_add_box(root, Vector3(0.62, 0.30, wall_t), Vector3(0.0, 0.15, -0.296), box_col, 0.88)
			_add_box(root, Vector3(0.62, 0.30, wall_t), Vector3(0.0, 0.15, 0.296), box_col, 0.88)
			_add_box(root, Vector3(wall_t, 0.30, 0.564), Vector3(-0.296, 0.15, 0.0), box_col, 0.88)
			_add_box(root, Vector3(wall_t, 0.30, 0.564), Vector3(0.296, 0.15, 0.0), box_col, 0.88)
			_object_bounds[obj_id] = {"offset": Vector3(0.0, 0.22, 0.0), "radius": 0.48}

		"high_shelf":
			# Multi-tier modern studio bookshelf with perch top at y = 1.80
			var wood := Color(0.70, 0.60, 0.49)
			var side_col := Color(0.64, 0.54, 0.44)
			_add_box(root, Vector3(0.045, elev, 0.42), Vector3(-0.42, elev * 0.5, 0.0), side_col, 0.78)
			_add_box(root, Vector3(0.045, elev, 0.42), Vector3(0.42, elev * 0.5, 0.0), side_col, 0.78)
			for shelf_y in [0.12, 0.68, 1.24, elev - 0.022]:
				_add_box(root, Vector3(0.88, 0.044, 0.42), Vector3(0.0, shelf_y, 0.0), wood, 0.75)
			# Decorative studio books
			_add_box(root, Vector3(0.08, 0.28, 0.24), Vector3(-0.24, 0.84, 0.0), Color(0.32, 0.46, 0.62), 0.75)
			_add_box(root, Vector3(0.07, 0.24, 0.22), Vector3(-0.15, 0.82, 0.0), Color(0.76, 0.44, 0.36), 0.75)
			_add_box(root, Vector3(0.09, 0.26, 0.24), Vector3(0.18, 1.39, 0.0), Color(0.38, 0.58, 0.50), 0.75)
			_object_bounds[obj_id] = {"offset": Vector3(0.0, elev * 0.55, 0.0), "radius": 0.78}

		"scratching_post":
			_add_box(root, Vector3(0.44, 0.06, 0.44), Vector3(0.0, 0.03, 0.0), Color(0.56, 0.50, 0.44), 0.86)
			_add_cylinder(root, 0.075, 0.72, Vector3(0.0, 0.39, 0.0), Color(0.80, 0.71, 0.53))
			_add_cylinder(root, 0.14, 0.04, Vector3(0.0, 0.76, 0.0), Color(0.62, 0.56, 0.50))
			_object_bounds[obj_id] = {"offset": Vector3(0.0, 0.40, 0.0), "radius": 0.38}

		"user":
			# Clean architectural human figure with floor indicator ring
			var base_ring := _add_cylinder(
				root, 0.29, 0.012, Vector3(0.0, 0.008, 0.0), Color(0.22, 0.46, 0.72, 0.38), true
			)
			base_ring.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			_add_cylinder(root, 0.20, 0.96, Vector3(0.0, 0.50, 0.0), Color(0.24, 0.38, 0.56))
			_add_box(root, Vector3(0.44, 0.14, 0.22), Vector3(0.0, 0.96, 0.0), Color(0.22, 0.35, 0.52), 0.72)
			var head := MeshInstance3D.new()
			var sph := SphereMesh.new()
			sph.radius = 0.155
			sph.height = 0.31
			head.mesh = sph
			var mat := StandardMaterial3D.new()
			mat.albedo_color = Color(0.89, 0.77, 0.67)
			mat.roughness = 0.68
			head.material_override = mat
			head.position = Vector3(0.0, 1.22, 0.0)
			root.add_child(head)
			_object_bounds[obj_id] = {"offset": Vector3(0.0, 0.68, 0.0), "radius": 0.48}

		_:
			_add_box(root, Vector3(0.35, 0.35, 0.35), Vector3(0.0, 0.18, 0.0), Color(0.6, 0.6, 0.6), 0.8)
			_object_bounds[obj_id] = {"offset": Vector3(0.0, 0.25, 0.0), "radius": 0.35}

	# Crisp floating architectural label
	var lbl := Label3D.new()
	lbl.text = str(data.get("name", obj_id.replace("_", " ")))
	lbl.font_size = 24
	lbl.pixel_size = 0.0036
	lbl.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	lbl.modulate = Color(0.18, 0.22, 0.28, 0.88)
	lbl.outline_size = 6
	lbl.outline_modulate = Color(1.0, 1.0, 1.0, 0.88)
	var label_y := maxf(elev + 0.52, 0.62)
	if obj_id == "user":
		label_y = 1.54
	elif obj_id == "water_glass":
		label_y = elev + 0.34
	lbl.position = Vector3(0.0, label_y, 0.0)
	root.add_child(lbl)
	return root


func _add_box(
	parent: Node3D,
	sz: Vector3,
	pos: Vector3,
	col: Color,
	roughness: float = 0.76,
) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var mesh := BoxMesh.new()
	mesh.size = sz
	mi.mesh = mesh
	var mat := StandardMaterial3D.new()
	mat.albedo_color = col
	mat.roughness = roughness
	mi.material_override = mat
	mi.position = pos
	parent.add_child(mi)
	return mi


func _add_cylinder(
	parent: Node3D,
	radius: float,
	height: float,
	pos: Vector3,
	col: Color,
	alpha: bool = false,
) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var mesh := CylinderMesh.new()
	mesh.top_radius = radius
	mesh.bottom_radius = radius
	mesh.height = height
	mi.mesh = mesh
	var mat := StandardMaterial3D.new()
	mat.albedo_color = col
	mat.roughness = 0.35 if alpha else 0.78
	if alpha:
		mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mi.material_override = mat
	mi.position = pos
	parent.add_child(mi)
	return mi
