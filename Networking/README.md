# Strict.Networking

TCP/IP sockets for the programs of the repository: a client connects to a host,
a server listens on a port and accepts, both send and receive bytes.

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
