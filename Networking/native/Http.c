/* HTTP and HTTPS requests: method, url, body and request header lines ("Name: value", one per line) give one text,
   "<status>\n<header lines, each ending in \n>\n<body>" (Http.Response reads it). Windows uses WinHTTP, Linux and
   macOS libcurl (loaded when first used), the web a synchronous XMLHttpRequest; the OS or curl checks certificates.
   Bodies are texts: their bytes (UTF-8) pass through unchanged. Failures: InvalidHttpUrl, HttpResponseTooLarge (more
   than maximum bytes) and HttpFailed with what the platform reported. */
#define _CRT_SECURE_NO_WARNINGS
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "strict.h"

static _Thread_local char* returned;

static const char* text_of(double value)
{
	const char* text = strict_text(value);
	return text ? text : "";
}

/* the Error keeps the name it is given, so every failure with details gets its own (failures are rare) */
static double failed(const char* format, ...)
{
	char* name = malloc(320);
	if (!name)
		return strict_fail("HttpFailed");
	va_list arguments;
	va_start(arguments, format);
	vsnprintf(name, 320, format, arguments);
	va_end(arguments);
	return strict_fail(name);
}

static int valid_url(const char* url)
{
	size_t scheme = strncmp(url, "https://", 8) == 0 ? 8 : strncmp(url, "http://", 7) == 0 ? 7 : 0;
	return scheme && url[scheme] && url[scheme] != '/' && url[scheme] != '?' && url[scheme] != ':';
}

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

/* the browser decides about timeouts, proxies and cross origin requests itself; the script must not contain escapes
   like \n, Emscripten turns them into the characters before the script is parsed */
EM_JS_DEPS(http, "$stringToNewUTF8,$UTF8ToString");
EM_JS(char*, http_requested, (const char* method, const char* url, const char* body, const char* lines), {
	try
	{
		var newline = String.fromCharCode(10);
		var request = new XMLHttpRequest();
		request.open(UTF8ToString(method), UTF8ToString(url), false);
		UTF8ToString(lines).split(newline).forEach(function(line)
		{
			var colon = line.indexOf(":");
			if (colon > 0)
				request.setRequestHeader(line.substring(0, colon).trim(), line.substring(colon + 1).trim());
		});
		var content = UTF8ToString(body);
		request.send(content.length ? content : null);
		var received = request.getAllResponseHeaders().split(String.fromCharCode(13)).join("");
		return stringToNewUTF8(request.status + newline + received + newline + request.responseText);
	}
	catch (error)
	{
		return stringToNewUTF8("!" + error);
	}
});

static double requested(const char* method, const char* url, const char* body, const char* lines, double seconds,
	double maximum)
{
	char* text = http_requested(method, url, body, lines);
	if (!text)
		return strict_fail("NotEnoughMemory");
	if (text[0] == '!')
	{
		double result = failed("HttpFailed (%s)", text + 1);
		free(text);
		return result;
	}
	char* separator = strstr(text, "\n\n");
	if (separator && strlen(separator + 2) > (size_t)maximum)
	{
		free(text);
		return strict_fail("HttpResponseTooLarge");
	}
	free(returned);
	returned = text;
	return strict_text_result(text);
}

#else

typedef struct
{
	char* data;
	size_t used;
	size_t capacity;
	size_t maximum;
	int exceeded;
} collected;

static int appended(collected* target, const char* data, size_t size)
{
	if (size > target->maximum - target->used)
	{
		target->exceeded = 1;
		return 0;
	}
	if (target->used + size + 1 > target->capacity)
	{
		size_t capacity = (target->used + size + 1) * 2;
		char* grown = realloc(target->data, capacity);
		if (!grown)
			return 0;
		target->data = grown;
		target->capacity = capacity;
	}
	memcpy(target->data + target->used, data, size);
	target->used += size;
	target->data[target->used] = 0;
	return 1;
}

static double responded(long status, const collected* headers, const collected* content)
{
	size_t size = 32 + headers->used + content->used;
	char* text = malloc(size);
	if (!text)
		return strict_fail("NotEnoughMemory");
	size_t written = (size_t)snprintf(text, size, "%ld\n", status);
	if (headers->used)
		memcpy(text + written, headers->data, headers->used);
	written += headers->used;
	text[written++] = '\n';
	if (content->used)
		memcpy(text + written, content->data, content->used);
	text[written + content->used] = 0;
	free(returned);
	returned = text;
	return strict_text_result(text);
}

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <wchar.h>
#include <windows.h>
#include <winhttp.h>
#ifndef WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_3
#define WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_3 0x00002000
#endif

typedef struct
{
	HINTERNET session, connection, request;
	wchar_t *address, *verb, *lines, *host, *path;
	INTERNET_PORT port;
	DWORD secure;
} windows_request;

static wchar_t* wide(const char* text)
{
	int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, -1, NULL, 0);
	wchar_t* result = count > 0 ? malloc((size_t)count * sizeof(wchar_t)) : NULL;
	if (result && !MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, -1, result, count))
	{
		free(result);
		return NULL;
	}
	return result;
}

/* WinHTTP wants the request header lines separated by \r\n */
static wchar_t* header_lines(const char* lines)
{
	char* separated = malloc(strlen(lines) * 2 + 1);
	if (!separated)
		return NULL;
	size_t written = 0;
	for (const char* next = lines; *next; next++)
	{
		if (*next == '\n' && (next == lines || next[-1] != '\r'))
			separated[written++] = '\r';
		separated[written++] = *next;
	}
	separated[written] = 0;
	wchar_t* result = wide(separated);
	free(separated);
	return result;
}

/* Windows Server 2012 has no automatic proxy access type and its WinHTTP defaults to TLS 1.0, servers need 1.2 */
static HINTERNET opened(void)
{
	HINTERNET session = WinHttpOpen(L"Strict", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, NULL, NULL, 0);
	if (!session)
		session = WinHttpOpen(L"Strict", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME,
			WINHTTP_NO_PROXY_BYPASS, 0);
	DWORD protocols = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2 | WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_3;
	if (session && !WinHttpSetOption(session, WINHTTP_OPTION_SECURE_PROTOCOLS, &protocols, sizeof(protocols)))
	{
		protocols = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2;
		if (!WinHttpSetOption(session, WINHTTP_OPTION_SECURE_PROTOCOLS, &protocols, sizeof(protocols)))
		{
			WinHttpCloseHandle(session);
			session = NULL;
		}
	}
	return session;
}

static double windows_failure(const char* call)
{
	return failed("HttpFailed (Windows error %lu in %s)", (unsigned long)GetLastError(), call);
}

/* the header lines of the response without its status line, WinHTTP gives them with \r\n and an empty last line */
static int windows_headers(HINTERNET request, collected* headers)
{
	DWORD size = 0;
	WinHttpQueryHeaders(request, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, WINHTTP_NO_OUTPUT_BUFFER,
		&size, WINHTTP_NO_HEADER_INDEX);
	wchar_t* raw = GetLastError() == ERROR_INSUFFICIENT_BUFFER ? malloc(size + sizeof(wchar_t)) : NULL;
	if (!raw || !WinHttpQueryHeaders(request, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, raw, &size,
		WINHTTP_NO_HEADER_INDEX))
	{
		free(raw);
		return 0;
	}
	raw[size / sizeof(wchar_t)] = 0;
	int bytes = WideCharToMultiByte(CP_UTF8, 0, raw, -1, NULL, 0, NULL, NULL);
	char* text = bytes > 0 ? malloc((size_t)bytes) : NULL;
	int converted = text && WideCharToMultiByte(CP_UTF8, 0, raw, -1, text, bytes, NULL, NULL);
	free(raw);
	char* line = converted ? strstr(text, "\r\n") : NULL;
	int kept = converted;
	while (kept && line && line[2])
	{
		char* start = line + 2;
		line = strstr(start, "\r\n");
		size_t length = line ? (size_t)(line - start) : strlen(start);
		kept = length == 0 || (appended(headers, start, length) && appended(headers, "\n", 1));
	}
	free(text);
	return kept;
}

static double windows_sent(windows_request* request, const char* body, DWORD milliseconds, DWORD* status,
	collected* headers, collected* content)
{
	request->session = opened();
	if (!request->session || !WinHttpSetTimeouts(request->session, milliseconds, milliseconds, milliseconds,
		milliseconds))
		return windows_failure("WinHttpOpen");
	request->connection = WinHttpConnect(request->session, request->host, request->port, 0);
	if (!request->connection)
		return windows_failure("WinHttpConnect");
	request->request = WinHttpOpenRequest(request->connection, request->verb, request->path, NULL, WINHTTP_NO_REFERER,
		WINHTTP_DEFAULT_ACCEPT_TYPES, request->secure);
	if (!request->request)
		return windows_failure("WinHttpOpenRequest");
	DWORD size = (DWORD)strlen(body);
	if (!WinHttpSendRequest(request->request, request->lines[0] ? request->lines : WINHTTP_NO_ADDITIONAL_HEADERS,
		(DWORD)-1L, size ? (void*)body : WINHTTP_NO_REQUEST_DATA, size, size, 0))
		return windows_failure("WinHttpSendRequest");
	DWORD statusSize = sizeof(*status);
	if (!WinHttpReceiveResponse(request->request, NULL) || !WinHttpQueryHeaders(request->request,
		WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, NULL, status, &statusSize, NULL))
		return windows_failure("WinHttpReceiveResponse");
	if (!windows_headers(request->request, headers))
		return headers->exceeded ? strict_fail("HttpResponseTooLarge") : windows_failure("WinHttpQueryHeaders");
	char chunk[65536];
	for (;;)
	{
		DWORD received = 0;
		if (!WinHttpReadData(request->request, chunk, sizeof(chunk), &received))
			return windows_failure("WinHttpReadData");
		if (received == 0)
			return 0;
		if (!appended(content, chunk, received))
			return content->exceeded ? strict_fail("HttpResponseTooLarge") : strict_fail("NotEnoughMemory");
	}
}

/* splits the url into host and path with query, only http and https with a host are accepted */
static int windows_parsed(windows_request* request, const char* url)
{
	URL_COMPONENTS parts = { 0 };
	parts.dwStructSize = sizeof(parts);
	parts.dwHostNameLength = parts.dwUrlPathLength = parts.dwExtraInfoLength = (DWORD)-1;
	request->address = wide(url);
	if (!request->address || !WinHttpCrackUrl(request->address, 0, 0, &parts) || parts.dwHostNameLength == 0 ||
		(parts.nScheme != INTERNET_SCHEME_HTTP && parts.nScheme != INTERNET_SCHEME_HTTPS))
		return 0;
	request->host = calloc((size_t)parts.dwHostNameLength + 1, sizeof(wchar_t));
	request->path = calloc((size_t)parts.dwUrlPathLength + parts.dwExtraInfoLength + 2, sizeof(wchar_t));
	if (!request->host || !request->path)
		return 0;
	wmemcpy(request->host, parts.lpszHostName, parts.dwHostNameLength);
	wmemcpy(request->path, parts.lpszUrlPath, parts.dwUrlPathLength);
	wmemcpy(request->path + parts.dwUrlPathLength, parts.lpszExtraInfo, parts.dwExtraInfoLength);
	request->port = parts.nPort;
	request->secure = parts.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0;
	return 1;
}

static double requested(const char* method, const char* url, const char* body, const char* lines, double seconds,
	double maximum)
{
	windows_request request = { 0 };
	collected headers = { 0 }, content = { 0 };
	headers.maximum = content.maximum = (size_t)maximum;
	DWORD status = 0;
	double result = windows_parsed(&request, url) ? 0 : strict_fail("InvalidHttpUrl");
	request.verb = wide(method);
	request.lines = header_lines(lines);
	if (strict_bits(result) == 0 && (!request.verb || !request.lines))
		result = strict_fail("InvalidHttpText");
	if (strict_bits(result) == 0)
		result = windows_sent(&request, body, (DWORD)(seconds * 1000), &status, &headers, &content);
	if (strict_bits(result) == 0)
		result = responded((long)status, &headers, &content);
	if (request.request)
		WinHttpCloseHandle(request.request);
	if (request.connection)
		WinHttpCloseHandle(request.connection);
	if (request.session)
		WinHttpCloseHandle(request.session);
	free(request.address);
	free(request.verb);
	free(request.lines);
	free(request.host);
	free(request.path);
	free(headers.data);
	free(content.data);
	return result;
}

#else
#include <dlfcn.h>
#include <pthread.h>

/* the few libcurl functions and numbers used, stable in its ABI for many years: loading it when first used builds a
   program without curl headers, macOS ships libcurl and Linux nearly always has libcurl.so.4 */
#define CURLOPT_WRITEDATA 10001
#define CURLOPT_URL 10002
#define CURLOPT_TIMEOUT 13
#define CURLOPT_POSTFIELDS 10015
#define CURLOPT_HTTPHEADER 10023
#define CURLOPT_HEADERDATA 10029
#define CURLOPT_CUSTOMREQUEST 10036
#define CURLOPT_NOBODY 44
#define CURLOPT_FOLLOWLOCATION 52
#define CURLOPT_POSTFIELDSIZE 60
#define CURLOPT_MAXREDIRS 68
#define CURLOPT_CONNECTTIMEOUT 78
#define CURLOPT_NOSIGNAL 99
#define CURLOPT_WRITEFUNCTION 20011
#define CURLOPT_HEADERFUNCTION 20079
#define CURLINFO_RESPONSE_CODE 0x200002
#define CURL_GLOBAL_DEFAULT 3

typedef struct curl_slist curl_slist;
typedef size_t (*curl_received)(char*, size_t, size_t, void*);
static struct
{
	void* (*easy_init)(void);
	int (*easy_setopt)(void*, int, ...);
	int (*easy_perform)(void*);
	int (*easy_getinfo)(void*, int, ...);
	void (*easy_cleanup)(void*);
	curl_slist* (*slist_append)(curl_slist*, const char*);
	void (*slist_free_all)(curl_slist*);
	const char* (*easy_strerror)(int);
	int ready;
} curl;
static pthread_once_t curl_once = PTHREAD_ONCE_INIT;

static void curl_loaded(void)
{
	const char* names[] = { "libcurl.so.4", "libcurl.so", "/usr/lib/libcurl.4.dylib", "libcurl.4.dylib" };
	void* library = NULL;
	for (size_t index = 0; !library && index < sizeof(names) / sizeof(names[0]); index++)
		library = dlopen(names[index], RTLD_NOW);
	if (!library)
		return;
	int (*global_init)(long) = (int (*)(long))dlsym(library, "curl_global_init");
	curl.easy_init = (void* (*)(void))dlsym(library, "curl_easy_init");
	curl.easy_setopt = (int (*)(void*, int, ...))dlsym(library, "curl_easy_setopt");
	curl.easy_perform = (int (*)(void*))dlsym(library, "curl_easy_perform");
	curl.easy_getinfo = (int (*)(void*, int, ...))dlsym(library, "curl_easy_getinfo");
	curl.easy_cleanup = (void (*)(void*))dlsym(library, "curl_easy_cleanup");
	curl.slist_append = (curl_slist* (*)(curl_slist*, const char*))dlsym(library, "curl_slist_append");
	curl.slist_free_all = (void (*)(curl_slist*))dlsym(library, "curl_slist_free_all");
	curl.easy_strerror = (const char* (*)(int))dlsym(library, "curl_easy_strerror");
	curl.ready = global_init && curl.easy_init && curl.easy_setopt && curl.easy_perform && curl.easy_getinfo &&
		curl.easy_cleanup && curl.slist_append && curl.slist_free_all && curl.easy_strerror &&
		global_init(CURL_GLOBAL_DEFAULT) == 0;
}

static size_t body_received(char* data, size_t size, size_t count, void* target)
{
	return appended(target, data, size * count) ? size * count : 0;
}

/* a status line starts the headers of another response (a redirect, 100 Continue): only the last ones are kept */
static size_t header_received(char* data, size_t size, size_t count, void* target)
{
	collected* headers = target;
	size_t length = size * count, line = length;
	if (length >= 5 && memcmp(data, "HTTP/", 5) == 0)
	{
		headers->used = 0;
		return length;
	}
	while (line > 0 && (data[line - 1] == '\n' || data[line - 1] == '\r'))
		line--;
	if (line == 0)
		return length;
	return appended(headers, data, line) && appended(headers, "\n", 1) ? length : 0;
}

static curl_slist* curl_lines(const char* lines)
{
	curl_slist* list = NULL;
	char *copy = strdup(lines), *rest = NULL;
	for (char* line = copy ? strtok_r(copy, "\r\n", &rest) : NULL; line; line = strtok_r(NULL, "\r\n", &rest))
		list = curl.slist_append(list, line);
	free(copy);
	return list;
}

static double curl_sent(void* handle, const char* method, const char* body, long seconds, collected* headers,
	collected* content)
{
	if (strcmp(method, "HEAD") == 0)
		curl.easy_setopt(handle, CURLOPT_NOBODY, 1L);
	else if (strcmp(method, "GET") != 0 || body[0])
	{
		curl.easy_setopt(handle, CURLOPT_POSTFIELDSIZE, (long)strlen(body));
		curl.easy_setopt(handle, CURLOPT_POSTFIELDS, body);
	}
	curl.easy_setopt(handle, CURLOPT_CUSTOMREQUEST, method);
	curl.easy_setopt(handle, CURLOPT_TIMEOUT, seconds);
	curl.easy_setopt(handle, CURLOPT_CONNECTTIMEOUT, seconds);
	curl.easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
	curl.easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
	curl.easy_setopt(handle, CURLOPT_MAXREDIRS, 10L);
	curl.easy_setopt(handle, CURLOPT_WRITEFUNCTION, (curl_received)body_received);
	curl.easy_setopt(handle, CURLOPT_WRITEDATA, content);
	curl.easy_setopt(handle, CURLOPT_HEADERFUNCTION, (curl_received)header_received);
	curl.easy_setopt(handle, CURLOPT_HEADERDATA, headers);
	int code = curl.easy_perform(handle);
	if (code == 0)
		return 0;
	return content->exceeded || headers->exceeded ? strict_fail("HttpResponseTooLarge") :
		failed("HttpFailed (curl error %d: %s)", code, curl.easy_strerror(code));
}

static double requested(const char* method, const char* url, const char* body, const char* lines, double seconds,
	double maximum)
{
	pthread_once(&curl_once, curl_loaded);
	void* handle = curl.ready ? curl.easy_init() : NULL;
	if (!handle)
		return strict_fail("HttpFailed (libcurl.so.4 or libcurl.4.dylib not found)");
	collected headers = { 0 }, content = { 0 };
	headers.maximum = content.maximum = (size_t)maximum;
	curl_slist* list = curl_lines(lines);
	curl.easy_setopt(handle, CURLOPT_URL, url);
	curl.easy_setopt(handle, CURLOPT_HTTPHEADER, list);
	double result = curl_sent(handle, method, body, (long)seconds, &headers, &content);
	long status = 0;
	if (strict_bits(result) == 0)
		result = curl.easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &status) == 0 ?
			responded(status, &headers, &content) : strict_fail("HttpFailed (curl_easy_getinfo)");
	curl.easy_cleanup(handle);
	curl.slist_free_all(list);
	free(headers.data);
	free(content.data);
	return result;
}
#endif
#endif

STRICT_EXPORT double Http_Requested(double self, double method, double url, double body, double headerLines,
	double seconds, double maximum)
{
	return valid_url(text_of(url)) ? requested(text_of(method), text_of(url), text_of(body), text_of(headerLines),
		seconds, maximum) : strict_fail("InvalidHttpUrl");
}

/* Started requests run on threads of their own (on the web right away, it has no threads), Polled gives "0\n\n" while
   one runs and its result once. A request number is its slot plus 64 times the slot's earlier uses, an old number never
   gets the response of a newer request. */
#define REQUESTS 64
static struct
{
	char *method, *url, *body, *lines, *text;
	double seconds, maximum, result;
	uint64_t uses;
	int state;
} requests[REQUESTS];

#ifdef _WIN32
static SRWLOCK guard = SRWLOCK_INIT;
static void locked(void) { AcquireSRWLockExclusive(&guard); }
static void unlocked(void) { ReleaseSRWLockExclusive(&guard); }
#elif !defined(__EMSCRIPTEN__)
static pthread_mutex_t guard = PTHREAD_MUTEX_INITIALIZER;
static void locked(void) { pthread_mutex_lock(&guard); }
static void unlocked(void) { pthread_mutex_unlock(&guard); }
#else
static void locked(void) {}
static void unlocked(void) {}
#endif

static void released(size_t slot)
{
	free(requests[slot].method);
	free(requests[slot].url);
	free(requests[slot].body);
	free(requests[slot].lines);
}

static void worked(size_t slot)
{
	free(returned);
	returned = NULL;
	double result = requested(requests[slot].method, requests[slot].url, requests[slot].body, requests[slot].lines,
		requests[slot].seconds, requests[slot].maximum);
	released(slot);
	locked();
	requests[slot].result = result;
	requests[slot].text = returned;
	requests[slot].state = 2;
	unlocked();
	returned = NULL;
}

#ifdef _WIN32
static DWORD WINAPI working(void* slot) { worked((size_t)slot); return 0; }
static int thread_started(size_t slot)
{
	HANDLE thread = CreateThread(0, 0, working, (void*)slot, 0, 0);
	if (!thread)
		return 0;
	CloseHandle(thread);
	return 1;
}
#elif !defined(__EMSCRIPTEN__)
static void* working(void* slot) { worked((size_t)slot); return 0; }
static int thread_started(size_t slot)
{
	pthread_t thread;
	if (pthread_create(&thread, 0, working, (void*)slot) != 0)
		return 0;
	pthread_detach(thread);
	return 1;
}
#else
static int thread_started(size_t slot) { worked(slot); return 1; }
#endif

STRICT_EXPORT double Http_Started(double self, double method, double url, double body, double headerLines,
	double seconds, double maximum)
{
	if (!valid_url(text_of(url)))
		return strict_fail("InvalidHttpUrl");
	locked();
	size_t slot = 0;
	while (slot < REQUESTS && requests[slot].state)
		slot++;
	if (slot < REQUESTS)
		requests[slot].state = 1;
	unlocked();
	if (slot == REQUESTS)
		return strict_fail("TooManyHttpRequests");
	requests[slot].method = strdup(text_of(method));
	requests[slot].url = strdup(text_of(url));
	requests[slot].body = strdup(text_of(body));
	requests[slot].lines = strdup(text_of(headerLines));
	requests[slot].seconds = seconds;
	requests[slot].maximum = maximum;
	double number = (double)(requests[slot].uses * REQUESTS + slot + 1);
	int copied = requests[slot].method && requests[slot].url && requests[slot].body && requests[slot].lines;
	if (copied && thread_started(slot))
		return number;
	released(slot);
	locked();
	requests[slot].state = 0;
	unlocked();
	return strict_fail(copied ? "HttpFailed (no thread for the request)" : "NotEnoughMemory");
}

STRICT_EXPORT double Http_Polled(double self, double request)
{
	uint64_t number = request >= 1 && request < 1e15 ? (uint64_t)request : 0;
	size_t slot = number ? (size_t)((number - 1) % REQUESTS) : 0;
	locked();
	int state = number && (double)number == request && requests[slot].uses == (number - 1) / REQUESTS ?
		requests[slot].state : 0;
	double result = state == 0 ? strict_fail("InvalidHttpRequest") : state == 1 ? strict_text_result("0\n\n") :
		requests[slot].result;
	if (state == 2)
	{
		free(returned);
		returned = requests[slot].text;
		requests[slot].state = 0;
		requests[slot].uses++;
	}
	unlocked();
	return result;
}
