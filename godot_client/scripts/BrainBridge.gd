class_name BrainBridge
extends Node

signal state_received(state: Dictionary)
signal connection_changed(is_connected: bool)
signal archive_loaded(elites: Array, metadata: Dictionary)
signal archive_error(message: String)

@export var server_url: String = "ws://127.0.0.1:8765/ws"
@export var reconnect_delay_sec: float = 2.0

var _socket: WebSocketPeer = WebSocketPeer.new()
var _http: HTTPRequest = null
var _connected: bool = false
var _reconnect_timer: float = 0.0
var _archive_in_flight: bool = false


func _ready() -> void:
	_http = HTTPRequest.new()
	_http.timeout = 8.0
	add_child(_http)
	_http.request_completed.connect(_on_archive_request_completed)
	connect_to_brain()


func is_connected_to_brain() -> bool:
	return _connected and _socket != null and _socket.get_ready_state() == WebSocketPeer.STATE_OPEN


func get_http_base_url() -> String:
	var url := server_url.strip_edges()
	var query_idx := url.find("?")
	if query_idx != -1:
		url = url.substr(0, query_idx)
	if url.begins_with("wss://"):
		url = "https://" + url.substr(6)
	elif url.begins_with("ws://"):
		url = "http://" + url.substr(5)
	if url.ends_with("/ws"):
		url = url.substr(0, url.length() - 3)
	while url.ends_with("/"):
		url = url.substr(0, url.length() - 1)
	return url


func connect_to_brain() -> void:
	if _socket != null:
		var current_state := _socket.get_ready_state()
		if current_state == WebSocketPeer.STATE_CONNECTING or current_state == WebSocketPeer.STATE_CLOSING:
			return
	_socket = WebSocketPeer.new()
	_reconnect_timer = reconnect_delay_sec
	var err := _socket.connect_to_url(server_url)
	if err != OK:
		_set_connected(false)


func _process(delta: float) -> void:
	if _socket == null:
		return
	_socket.poll()
	var state := _socket.get_ready_state()

	if state == WebSocketPeer.STATE_OPEN:
		if not _connected:
			_set_connected(true)
			fetch_archive()
		while _socket.get_available_packet_count() > 0:
			var raw := _socket.get_packet().get_string_from_utf8()
			var parsed: Variant = JSON.parse_string(raw)
			if parsed is Dictionary:
				var frame: Dictionary = parsed
				if not frame.has("error"):
					state_received.emit(frame)
				else:
					push_warning("BrainBridge server error: %s" % str(frame.get("error", "")))
	elif state == WebSocketPeer.STATE_CLOSING:
		if _connected:
			_set_connected(false)
	elif state == WebSocketPeer.STATE_CLOSED:
		if _connected:
			_set_connected(false)
		_reconnect_timer -= delta
		if _reconnect_timer <= 0.0:
			_reconnect_timer = reconnect_delay_sec
			connect_to_brain()


func send_command(payload: Dictionary) -> bool:
	if _socket == null or _socket.get_ready_state() != WebSocketPeer.STATE_OPEN:
		return false
	var err := _socket.send_text(JSON.stringify(payload))
	return err == OK


func fetch_archive() -> void:
	if _http == null or _archive_in_flight:
		return
	var archive_url := get_http_base_url() + "/archive"
	var err := _http.request(archive_url)
	if err != OK:
		archive_error.emit("Archive HTTP request failed (code %d)" % err)
		return
	_archive_in_flight = true


func _on_archive_request_completed(
	result: int,
	response_code: int,
	_headers: PackedStringArray,
	body: PackedByteArray,
) -> void:
	_archive_in_flight = false
	if result != HTTPRequest.RESULT_SUCCESS or response_code < 200 or response_code >= 300:
		archive_error.emit("Archive fetch failed (HTTP %d)" % response_code)
		return
	var text := body.get_string_from_utf8()
	var parsed: Variant = JSON.parse_string(text)
	if not (parsed is Dictionary):
		archive_error.emit("Invalid archive JSON payload")
		return
	var data: Dictionary = parsed
	var elites_var: Variant = data.get("elites", [])
	var meta_var: Variant = data.get("metadata", {})
	var elites: Array = elites_var if elites_var is Array else []
	var metadata: Dictionary = meta_var if meta_var is Dictionary else {}
	archive_loaded.emit(elites, metadata)


func _set_connected(value: bool) -> void:
	if _connected != value:
		_connected = value
		connection_changed.emit(value)
