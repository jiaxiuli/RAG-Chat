let ws;
export function connectWS(onMessage) {
  ws = new WebSocket('ws://localhost:8000/ws/chat');
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
}
export function sendWS(content, conversation_id) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'user_message', content, conversation_id }));
}
