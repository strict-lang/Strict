/* TCP/IP sockets: a client connects to a host, a server listens and accepts, both send and receive bytes.
   The handle of a socket is its number, 0 is no socket. Windows uses winsock2, the other systems BSD sockets. */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#ifndef _WIN32
#include <fcntl.h>
#endif
#include "strict.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#define close_socket closesocket
typedef int socket_size;
#else
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#define close_socket close
typedef socklen_t socket_size;
#endif

#ifdef _WIN32
static INIT_ONCE startup = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK started(PINIT_ONCE once, PVOID argument, PVOID* context)
{
	WSADATA data;
	return WSAStartup(MAKEWORD(2, 2), &data) == 0;
}
#endif

/* Windows sockets are made without the overlapped flag, so a Logger with this handle as output can WriteFile to it. */
static int open_socket(int family, int type, int protocol)
{
#ifdef _WIN32
	return (int)WSASocketW(family, type, protocol, 0, 0, 0);
#else
	return (int)socket(family, type, protocol);
#endif
}

/* Small writes (log lines of a resident server request) go out at once instead of waiting for the delayed ACK. */
static int sent_at_once(int handle)
{
	int flag = 1;
	setsockopt(handle, IPPROTO_TCP, TCP_NODELAY, (const char*)&flag, (socket_size)sizeof(flag));
	return handle;
}

static int ensure_started(void)
{
#ifdef _WIN32
	return InitOnceExecuteOnce(&startup, started, 0, 0);
#else
	return 1;
#endif
}

static int invalid(void)
{
#ifdef _WIN32
	return (int)INVALID_SOCKET;
#else
	return -1;
#endif
}

static int is_port(double port)
{
	return port >= 0 && port <= 65535 && port == (double)(int)port;
}

STRICT_EXPORT double Socket_Connect(double self, double host, double port)
{
	if (!is_port(port))
		return strict_fail("InvalidPort");
	if (!ensure_started())
		return strict_fail("SocketStartupFailed");
	char service[16];
	snprintf(service, sizeof(service), "%d", (int)port);
	struct addrinfo hints, *list = 0;
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	if (getaddrinfo(strict_text(host), service, &hints, &list) != 0 || list == 0)
		return strict_fail("HostNotFound");
	int handle = open_socket(list->ai_family, list->ai_socktype, list->ai_protocol);
	if (handle == invalid())
	{
		freeaddrinfo(list);
		return strict_fail("SocketFailed");
	}
	if (connect(handle, list->ai_addr, (socket_size)list->ai_addrlen) != 0)
	{
		close_socket(handle);
		freeaddrinfo(list);
		return strict_fail("ConnectionFailed");
	}
	freeaddrinfo(list);
	return (double)sent_at_once(handle);
}

/* A server on all interfaces (other computers can connect) or on 127.0.0.1 only. */
static double listening(double port, unsigned long host)
{
	if (!is_port(port))
		return strict_fail("InvalidPort");
	if (!ensure_started())
		return strict_fail("SocketStartupFailed");
	int handle = open_socket(AF_INET, SOCK_STREAM, 0);
	if (handle == invalid())
		return strict_fail("SocketFailed");
	int reuse = 1;
	setsockopt(handle, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, (socket_size)sizeof(reuse));
	struct sockaddr_in address;
	memset(&address, 0, sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(host);
	address.sin_port = htons((unsigned short)(int)port);
	if (bind(handle, (struct sockaddr*)&address, (socket_size)sizeof(address)) != 0 || listen(handle, SOMAXCONN) != 0)
	{
		close_socket(handle);
		return strict_fail("ListenFailed");
	}
	return (double)handle;
}

STRICT_EXPORT double Socket_Listen(double self, double port)
{
	return listening(port, INADDR_ANY);
}

STRICT_EXPORT double Socket_Local(double self, double port)
{
	return listening(port, INADDR_LOOPBACK);
}

STRICT_EXPORT double Socket_Port(double self, double socket)
{
	if ((int)socket <= 0)
		return strict_fail("InvalidSocket");
	struct sockaddr_in address;
	socket_size size = sizeof(address);
	if (getsockname((int)socket, (struct sockaddr*)&address, &size) != 0)
		return strict_fail("SocketAddressFailed");
	return ntohs(address.sin_port);
}

STRICT_EXPORT double Socket_Accept(double self, double server)
{
	if ((int)server <= 0)
		return strict_fail("InvalidSocket");
	int handle = (int)accept((int)server, 0, 0);
	if (handle == invalid())
		return strict_fail("AcceptFailed");
	return (double)sent_at_once(handle);
}

STRICT_EXPORT double Socket_Waiting(double self, double server, double seconds)
{
	if ((int)server <= 0)
		return strict_fail("InvalidSocket");
	fd_set readable;
	FD_ZERO(&readable);
	FD_SET((int)server, &readable);
	double wait = seconds > 0 ? seconds : 0;
	struct timeval limit = {(long)wait, (long)((wait - (long)wait) * 1000000)};
	return select((int)server + 1, &readable, 0, 0, &limit) > 0 ? 1 : 0;
}

STRICT_EXPORT double Socket_Send(double self, double socket, double bytes)
{
	if ((int)socket <= 0)
		return strict_fail("InvalidSocket");
	size_t count = (size_t)strict_count(bytes);
	if (count == 0)
		return 0;
	int isBytes = strict_format(bytes) == 9;
	unsigned char* buffer = isBytes ? (unsigned char*)strict_data(bytes) : (unsigned char*)malloc(count);
	if (buffer == 0)
		return strict_fail("OutOfMemory");
	for (size_t index = 0; !isBytes && index < count; index++)
		buffer[index] = (unsigned char)(int)strict_number(bytes, index);
	size_t sent = 0;
	while (sent < count)
	{
		int written = (int)send((int)socket, (const char*)buffer + sent, (int)(count - sent), 0);
		if (written <= 0)
		{
			if (!isBytes)
				free(buffer);
			return strict_fail("SendFailed");
		}
		sent += (size_t)written;
	}
	if (!isBytes)
		free(buffer);
	return 0;
}

STRICT_EXPORT double Socket_Receive(double self, double socket, double count)
{
	if ((int)socket <= 0)
		return strict_fail("InvalidSocket");
	int wanted = (int)count;
	if (wanted < 0)
		return strict_fail("InvalidCount");
	if (wanted == 0)
		return strict_numbers_result((const strict_numbers*)calloc(1, sizeof(strict_numbers)));
	unsigned char* buffer = (unsigned char*)malloc((size_t)wanted);
	if (buffer == 0)
		return strict_fail("OutOfMemory");
	int got = 0;
	while (got < wanted)
	{
		int read = (int)recv((int)socket, buffer + got, wanted - got, 0);
		if (read <= 0)
		{
			free(buffer);
			return strict_fail("ReceiveFailed");
		}
		got += read;
	}
	size_t size = sizeof(strict_numbers) + sizeof(double) * (size_t)wanted;
	strict_numbers* result = (strict_numbers*)malloc(size);
	if (result == 0)
	{
		free(buffer);
		return strict_fail("OutOfMemory");
	}
	result->count = wanted;
	for (int index = 0; index < wanted; index++)
		result->items[index] = (double)buffer[index];
	free(buffer);
	return strict_numbers_result(result);
}

STRICT_EXPORT double Socket_Pending(double self, double socket)
{
	if ((int)socket <= 0)
		return strict_fail("InvalidSocket");
	int count = 0;
#ifdef _WIN32
	u_long value = 0;
	if (ioctlsocket((SOCKET)socket, FIONREAD, &value) != 0)
		return strict_fail("PendingFailed");
	count = (int)value;
#else
	if (ioctl((int)socket, FIONREAD, &count) != 0)
		return strict_fail("PendingFailed");
#endif
	return (double)count;
}

STRICT_EXPORT double Socket_Close(double self, double socket)
{
	if ((int)socket > 0)
		close_socket((int)socket);
	return 0;
}

static int would_block(void)
{
#ifdef _WIN32
	return WSAGetLastError() == WSAEWOULDBLOCK;
#else
	return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINPROGRESS;
#endif
}

static int set_nonblocking(int handle, int enabled)
{
#ifdef _WIN32
	u_long mode = enabled ? 1 : 0;
	return ioctlsocket((SOCKET)handle, FIONBIO, &mode) == 0;
#else
	int flags = fcntl(handle, F_GETFL, 0);
	if (flags < 0)
		return 0;
	if (enabled)
		flags |= O_NONBLOCK;
	else
		flags &= ~O_NONBLOCK;
	return fcntl(handle, F_SETFL, flags) == 0;
#endif
}

/* Connect, but give up after seconds instead of waiting for the system timeout. */
STRICT_EXPORT double Socket_Try(double self, double host, double port, double seconds)
{
	if (!is_port(port) || seconds < 0)
		return strict_fail("InvalidPort");
	if (!ensure_started())
		return strict_fail("SocketStartupFailed");
	char service[16];
	snprintf(service, sizeof(service), "%d", (int)port);
	struct addrinfo hints, *list = 0;
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	if (getaddrinfo(strict_text(host), service, &hints, &list) != 0 || list == 0)
		return strict_fail("HostNotFound");
	int handle = open_socket(list->ai_family, list->ai_socktype, list->ai_protocol);
	if (handle == invalid() || !set_nonblocking(handle, 1))
	{
		if (handle != invalid())
			close_socket(handle);
		freeaddrinfo(list);
		return strict_fail("SocketFailed");
	}
	int connected = connect(handle, list->ai_addr, (socket_size)list->ai_addrlen) == 0;
	if (!connected && !would_block())
	{
		close_socket(handle);
		freeaddrinfo(list);
		return strict_fail("ConnectionFailed");
	}
	freeaddrinfo(list);
	if (!connected)
	{
		fd_set writable;
		FD_ZERO(&writable);
		FD_SET(handle, &writable);
		struct timeval limit = {(long)seconds, (long)((seconds - (long)seconds) * 1000000)};
		if (select(handle + 1, 0, &writable, 0, &limit) <= 0)
		{
			close_socket(handle);
			return strict_fail("ConnectionFailed");
		}
		int code = 0;
		socket_size size = sizeof(code);
		getsockopt(handle, SOL_SOCKET, SO_ERROR, (char*)&code, &size);
		if (code != 0)
		{
			close_socket(handle);
			return strict_fail("ConnectionFailed");
		}
	}
	set_nonblocking(handle, 0);
	return (double)sent_at_once(handle);
}

/* UDP socket on every interface, so a StrictGen can shout and hear on the local network. */
STRICT_EXPORT double Socket_Beacon(double self, double port)
{
	if (!is_port(port))
		return strict_fail("InvalidPort");
	if (!ensure_started())
		return strict_fail("SocketStartupFailed");
	int handle = open_socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (handle == invalid())
		return strict_fail("SocketFailed");
	int reuse = 1;
	int broadcast = 1;
	setsockopt(handle, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, (socket_size)sizeof(reuse));
	setsockopt(handle, SOL_SOCKET, SO_BROADCAST, (const char*)&broadcast, (socket_size)sizeof(broadcast));
	struct sockaddr_in address;
	memset(&address, 0, sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_ANY);
	address.sin_port = htons((unsigned short)(int)port);
	if (bind(handle, (struct sockaddr*)&address, (socket_size)sizeof(address)) != 0 || !set_nonblocking(handle, 1))
	{
		close_socket(handle);
		return strict_fail("ListenFailed");
	}
	return (double)handle;
}

static double told(int beacon, struct sockaddr_in* address, double bytes)
{
	if (beacon <= 0)
		return strict_fail("InvalidSocket");
	size_t count = (size_t)strict_count(bytes);
	unsigned char* buffer = count == 0 ? 0 : (unsigned char*)malloc(count);
	if (count > 0 && buffer == 0)
		return strict_fail("OutOfMemory");
	for (size_t index = 0; index < count; index++)
		buffer[index] = (unsigned char)(int)strict_number(bytes, index);
	int sent = (int)sendto(beacon, (const char*)buffer, (int)count, 0, (struct sockaddr*)address, (socket_size)sizeof(*address));
	free(buffer);
	return sent < 0 ? strict_fail("SendFailed") : 0;
}

STRICT_EXPORT double Socket_Tell(double self, double beacon, double host, double port, double bytes)
{
	if (!is_port(port))
		return strict_fail("InvalidPort");
	struct sockaddr_in address;
	memset(&address, 0, sizeof(address));
	address.sin_family = AF_INET;
	address.sin_port = htons((unsigned short)(int)port);
	if (inet_pton(AF_INET, strict_text(host), &address.sin_addr) != 1)
		return strict_fail("HostNotFound");
	return told((int)beacon, &address, bytes);
}

STRICT_EXPORT double Socket_Shout(double self, double beacon, double port, double bytes)
{
	if (!is_port(port))
		return strict_fail("InvalidPort");
	struct sockaddr_in address;
	memset(&address, 0, sizeof(address));
	address.sin_family = AF_INET;
	address.sin_port = htons((unsigned short)(int)port);
	address.sin_addr.s_addr = htonl(INADDR_BROADCAST);
	return told((int)beacon, &address, bytes);
}

static char heard_text[4096];

/* One datagram, or an empty text when nothing is waiting. "sender-ip\\tbody". */
STRICT_EXPORT double Socket_Hear(double self, double beacon)
{
	if ((int)beacon <= 0)
		return strict_fail("InvalidSocket");
	char buffer[2048];
	struct sockaddr_in sender;
	socket_size size = sizeof(sender);
	int got = (int)recvfrom((int)beacon, buffer, (int)sizeof(buffer) - 1, 0, (struct sockaddr*)&sender, &size);
	if (got < 0)
		return would_block() ? strict_text_result("") : strict_fail("ReceiveFailed");
	buffer[got] = 0;
	char ip[64];
	inet_ntop(AF_INET, &sender.sin_addr, ip, sizeof(ip));
	snprintf(heard_text, sizeof(heard_text), "%s\t%s", ip, buffer);
	return strict_text_result(heard_text);
}
