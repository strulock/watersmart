"""Root conftest — Windows asyncio socket workaround.

On Windows, asyncio's event loop uses socket.socketpair() internally for its
self-pipe. pytest-socket (pulled in by pytest-homeassistant-custom-component)
blocks all socket creation, which prevents the event loop from starting.

This root conftest captures the real socket class before pytest-socket patches
it, then replaces socket.socketpair with an implementation that bypasses the
block. This is Windows-only and has no effect on Linux/macOS CI.
"""

import sys

if sys.platform == "win32":
    import _socket
    import socket as _socket_module

    _RealSocket = _socket_module.socket

    def _socketpair_win(
        family=_socket_module.AF_INET,
        type=_socket_module.SOCK_STREAM,  # noqa: A002
        proto=0,
    ):
        """socketpair() that temporarily restores the real socket class.

        pytest-socket patches socket.socket with GuardedSocket. asyncio's
        ProactorEventLoop needs socket.socket subclasses (for weakrefs), so
        we can't use _socket.socket directly. Instead, swap back the real
        socket class for the duration of the pair creation.
        """
        import socket

        guarded = socket.socket
        socket.socket = _RealSocket
        try:
            listener = socket.socket(family, type, proto)
            try:
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                client = socket.socket(family, type, proto)
                try:
                    client.connect(listener.getsockname())
                    server, _ = listener.accept()
                except Exception:
                    client.close()
                    raise
            finally:
                listener.close()
        finally:
            socket.socket = guarded
        return server, client

    _socket_module.socketpair = _socketpair_win
