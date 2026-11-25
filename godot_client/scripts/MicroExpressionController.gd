class_name MicroExpressionController
extends Node

var _time: float = 0.0
var _blink_timer: float = 2.4


func update_expressions(
	pet: ProceduralPet,
	pad: Dictionary,
	look_target: Vector3,
	move_speed: float,
	delta: float,
) -> void:
	_time += delta
	var pleasure: float = float(pad.get("pleasure", 0.0))
	var arousal: float = float(pad.get("arousal", 0.0))
	var dominance: float = float(pad.get("dominance", 0.0))
	var fear: float = float(pad.get("fear_intensity", 0.0))
	var posture: String = pet.get_posture_state()

	var damp_fast := 1.0 - exp(-10.0 * delta)
	var damp_med := 1.0 - exp(-6.5 * delta)

	# 1. Posture-aware torso height, breathing, and spine pitch (in flip_pivot local space)
	var breath_rate := 1.8 if posture == "sleeping_curled" else (2.6 + maxf(0.0, arousal) * 2.4)
	var breath := sin(_time * breath_rate) * (0.010 if posture == "sleeping_curled" else 0.015)

	var target_torso_y := breath + fear * 0.045
	var target_torso_pitch := -fear * 0.20
	var target_torso_yaw := 0.0
	var target_head_offset_y := 0.105

	match posture:
		"sleeping_curled":
			target_torso_y = -0.095 + breath
			target_torso_pitch = 0.10
			target_torso_yaw = 0.22
			target_head_offset_y = 0.055
		"crouched":
			target_torso_y = -0.055 + breath + fear * 0.03
			target_torso_pitch = 0.14 - fear * 0.24
			target_head_offset_y = 0.085
		"sitting_alert":
			target_torso_y = 0.01 + breath
			target_torso_pitch = -0.14
			target_head_offset_y = 0.125
		"ceiling_cling":
			target_torso_y = -0.015 + breath * 0.7
			target_torso_pitch = 0.0
			target_head_offset_y = 0.105
		_:
			# walking
			var bob := absf(sin(_time * 9.5)) * 0.014 * clampf(move_speed, 0.0, 1.0)
			target_torso_y = bob + breath
			target_torso_pitch = -fear * 0.15

	pet.torso_pivot.position.y = lerpf(pet.torso_pivot.position.y, target_torso_y, damp_med)
	pet.torso_pivot.rotation.x = lerpf(pet.torso_pivot.rotation.x, target_torso_pitch, damp_med)
	pet.torso_pivot.rotation.y = lerpf(pet.torso_pivot.rotation.y, target_torso_yaw, damp_med)
	pet.head_pivot.position.y = lerpf(pet.head_pivot.position.y, target_head_offset_y, damp_med)

	# 2. Gravity-invariant head look-at tracking in torso_pivot local space
	if pet.torso_pivot.is_inside_tree():
		var local_target := pet.torso_pivot.to_local(look_target) - pet.head_pivot.position
		var yaw := clampf(atan2(local_target.x, local_target.z), -0.95, 0.95)
		var horiz := maxf(0.20, Vector2(local_target.x, local_target.z).length())
		var pitch := clampf(-atan2(local_target.y, horiz), -0.58, 0.58)
		if posture == "sleeping_curled":
			yaw *= 0.25
			pitch = 0.24
		pet.head_pivot.rotation.y = lerp_angle(pet.head_pivot.rotation.y, yaw, damp_fast)
		pet.head_pivot.rotation.x = lerp_angle(pet.head_pivot.rotation.x, pitch, damp_fast)

	# 3. Ear swivel & airplane-ear flattening when fearful or agitated
	var ear_flatten := clampf(fear * 0.85 + maxf(0.0, -pleasure) * 0.42, 0.0, 1.0)
	if posture == "sleeping_curled":
		ear_flatten = maxf(ear_flatten, 0.22)
	var ear_twitch := sin(_time * 11.0) * 0.05 * maxf(0.0, arousal)
	pet.left_ear.rotation.z = lerpf(pet.left_ear.rotation.z, 0.18 + ear_flatten * 0.72 + ear_twitch, damp_fast)
	pet.right_ear.rotation.z = lerpf(pet.right_ear.rotation.z, -(0.18 + ear_flatten * 0.72), damp_fast)

	# 4. Pupil dilation from arousal + periodic idle blinking + sleepy squint
	_blink_timer -= delta
	var blink_scale_y := 1.0
	if _blink_timer <= 0.0:
		if _blink_timer > -0.11:
			blink_scale_y = 0.12
		else:
			_blink_timer = randf_range(2.2, 4.8)

	var eye_open_y := 0.28 if posture == "sleeping_curled" else blink_scale_y
	if pet.left_eye_root != null and pet.right_eye_root != null:
		pet.left_eye_root.scale.y = lerpf(pet.left_eye_root.scale.y, eye_open_y, damp_fast)
		pet.right_eye_root.scale.y = lerpf(pet.right_eye_root.scale.y, eye_open_y, damp_fast)

	var pupil_size := clampf(0.65 + (arousal + 1.0) * 0.36, 0.55, 1.38)
	pet.left_pupil.scale = Vector3(pupil_size, pupil_size, 1.0)
	pet.right_pupil.scale = Vector3(pupil_size, pupil_size, 1.0)

	# 5. 5-segment procedural tail posture (curled when sleeping, high when happy, flicking when aroused)
	var tail_freq := 1.2 if posture == "sleeping_curled" else (2.2 + clampf((arousal + 1.0) * 3.2, 0.5, 8.0))
	var tail_amp := 0.04 if posture == "sleeping_curled" else (0.12 + absf(arousal) * 0.24)
	var tail_lift := clampf(0.45 + pleasure * 0.45 + dominance * 0.35 - fear * 0.85, -0.45, 1.15)
	for i in range(pet.tail_segments.size()):
		var seg: Node3D = pet.tail_segments[i]
		if posture == "sleeping_curled":
			seg.rotation.y = lerpf(seg.rotation.y, 0.46, damp_med)
			seg.rotation.x = lerpf(seg.rotation.x, 0.06, damp_med)
		else:
			var wave := sin(_time * tail_freq - float(i) * 0.58) * tail_amp
			seg.rotation.y = lerpf(seg.rotation.y, wave, damp_fast)
			var target_tx := -tail_lift * 0.34 if i == 0 else -tail_lift * 0.14
			seg.rotation.x = lerpf(seg.rotation.x, target_tx, damp_med)

	# 6. Posture-aware 4-legged walking gait / tuck / sit
	var stride := clampf(move_speed * 1.45, 0.0, 0.65)
	var gait_t := _time * 9.8
	for i in range(pet.leg_pivots.size()):
		var leg_pivot: Node3D = pet.leg_pivots[i]
		var target_rx := 0.0
		if move_speed > 0.08:
			var phase := 0.0 if (i == 0 or i == 3) else PI
			target_rx = sin(gait_t + phase) * stride
		elif posture == "sleeping_curled":
			target_rx = -1.15 if i < 2 else 1.10
		elif posture == "crouched":
			target_rx = -0.32 if i < 2 else 0.36
		elif posture == "sitting_alert":
			target_rx = 0.08 if i < 2 else 0.42
		leg_pivot.rotation.x = lerpf(leg_pivot.rotation.x, target_rx, damp_fast)
