# Strict.Networking

HTTP and HTTPS requests, HMAC-SHA256 signatures and TCP/IP sockets for the
programs of the repository.

- `Http.Send(method, url, body, headers, seconds, maximum)`: the `HttpResponse`
  (`status`, `headers` and `body`) of a request, any status is a response.
  `headers` are "Name: value" lines, `body` and `headers` default to empty,
  `seconds` (15) limits each step, `maximum` (64 MB) the bytes of the response.
  Errors are `InvalidHttpUrl` (only http and https), `HttpResponseTooLarge` and
  `HttpFailed (...)` with what the platform reported, like
  `HttpFailed (Windows error 12007 in WinHttpSendRequest)` for an unknown host.
- `Hmac.Sha256(secret, message)`: the HMAC-SHA256 of the message as lowercase
  hex, the signature of APIs like Binance, in portable C (`native/Hmac.c`).

Windows uses WinHTTP (TLS 1.2 and 1.3, the system proxy), Linux and macOS load
libcurl when first used (no headers needed to build), the web a synchronous
XMLHttpRequest. The OS or curl checks the certificates. Bodies are texts, their
UTF-8 bytes pass through unchanged. `Http.strict` tests the url checks and the
response format offline, `HttpChecks.strict` the signatures and (Slow) a request
to api.binance.com.

TCP/IP sockets: a client connects to a host, a server listens on a port and
accepts, both send and receive bytes.

- `Socket.Connect(host, port)`: the socket of a client, or an Error such as
  `HostNotFound` or `ConnectionFailed`.
- `Socket.Listen(port)`: the socket of a server; zero asks the OS for an available port.
- `Socket.Port(socket)`: its local port, including the port assigned by the OS.
- `Socket.Accept(server)`: the socket of the client that connects.
- `Socket.Send(socket, bytes)`: sends the bytes (0 to 255) and returns how many
  went out.
- `Socket.Receive(socket, count)`: reads exactly the bytes, they arrive in the
  order they were sent.
- `Socket.Pending(socket)`: the bytes that are already there, a game polls it
  to read a message only when it is complete.
- `Socket.Close(socket)`: closes the socket.

Windows uses winsock2, Linux and MacOS the BSD sockets, so the same program
runs on all of them. The native code is in `native/Socket.c`, the checks in
`SocketChecks.strict` open a server and a client on the loopback interface and
send bytes between them on OS-assigned ports. After building SocketChecks,
`python Networking/check_sockets.py` runs 16 copies of it at once.
