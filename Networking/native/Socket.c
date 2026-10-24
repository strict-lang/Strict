/* TCP/IP sockets: a client connects to a host, a server listens and accepts, both send and receive bytes.
   The handle of a socket is its number, 0 is no socket. Windows uses winsock2, the other systems BSD sockets. */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "strict.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#define close_socket closesocket
typedef int socket_size;
#else
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>
#define close_socket close
typedef socklen_t socket_size;
#endif

static int started = 0;

static void ensure_started(void)
{
	if (started)
		return;
#ifdef _WIN32
	WSADATA data;
	WSAStartup(MAKEWORD(2, 2), &data);
#endif
	started = 1;
}

static int invalid(void)
{
#ifdef _WIN32
	return (int)INVALID_SOCKET;
#else
	return -1;
#endif
}

STRICT_EXPORT double Socket_Connect(double self, double host, double port)
{
	ensure_started();
	char service[16];
	snprintf(service, sizeof(service), "%d", (int)port);
	struct addrinfo hints, *list = 0;
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	if (getaddrinfo(strict_text(host), service, &hints, &list) != 0 || list == 0)
		return strict_fail("HostNotFound");
	int handle = (int)socket(list->ai_family, list->ai_socktype, list->ai_protocol);
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
	return (double)handle;
}

STRICT_EXPORT double Socket_Listen(double self, double port)
{
	ensure_started();
	int handle = (int)socket(AF_INET, SOCK_STREAM, 0);
	if (handle == invalid())
		return strict_fail("SocketFailed");
	int reuse = 1;
	setsockopt(handle, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, (socket_size)sizeof(reuse));
	struct sockaddr_in address;
	memset(&address, 0, sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_ANY);
	address.sin_port = htons((unsigned short)(int)port);
	if (bind(handle, (struct sockaddr*)&address, (socket_size)sizeof(address)) != 0 || listen(handle, 8) != 0)
	{
		close_socket(handle);
		return strict_fail("ListenFailed");
	}
	return (double)handle;
}

STRICT_EXPORT double Socket_Accept(double self, double server)
{
	if ((int)server <= 0)
		return strict_fail("InvalidSocket");
	int handle = (int)accept((int)server, 0, 0);
	if (handle == invalid())
		return strict_fail("AcceptFailed");
	return (double)handle;
}

STRICT_EXPORT double Socket_Send(double self, double socket, double bytes)
{
	if ((int)socket <= 0)
		return strict_fail("InvalidSocket");
	const strict_numbers* data = strict_numbers_of(bytes);
	size_t count = (size_t)data->count;
	if (count == 0)
		return 0;
	unsigned char* buffer = (unsigned char*)malloc(count);
	if (buffer == 0)
		return strict_fail("OutOfMemory");
	for (size_t index = 0; index < count; index++)
		buffer[index] = (unsigned char)(int)data->items[index];
	size_t sent = 0;
	while (sent < count)
	{
		int written = (int)send((int)socket, (const char*)buffer + sent, (int)(count - sent), 0);
		if (written <= 0)
		{
			free(buffer);
			return strict_fail("SendFailed");
		}
		sent += (size_t)written;
	}
	free(buffer);
	return (double)sent;
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
	close_socket((int)socket);
	return 0;
}
