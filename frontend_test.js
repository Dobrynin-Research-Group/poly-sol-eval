var ws = null;
function connect(event) {
    var itemId = document.getElementById("itemId")
    var token = document.getElementById("token")
    var ws = new WebSocket("ws://localhost:8000/ws");
    ws.onmessage = function (event) {
        var messages = document.getElementById('messages')
        if (!messages) { return; }
        var message = document.createElement('li')
        var content = document.createTextNode(event.data)
        message.appendChild(content)
        messages.appendChild(message)
    };
    event.preventDefault()
}
function sendMessage(event) {
    var input = document.getElementById("messageText")
    if (ws != null && input != null) {
        ws.send(input.value)
        input.value = ''
        event.preventDefault()
    }
}