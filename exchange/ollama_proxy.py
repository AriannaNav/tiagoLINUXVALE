#!/usr/bin/env python3
"""ollama_proxy.py — forwards the container's localhost:11434 (where waiter.py
expects Ollama) to host.docker.internal:11434 (Ollama actually running on the
host machine). Needed because the container has no GPU access to run Ollama
itself; this just relays the TCP connection.

Run INSIDE the container (after Ollama is running on the host with the
qwen2.5:7b model pulled):
    python3 /root/ollama_proxy.py
"""
import asyncio

LISTEN_HOST, LISTEN_PORT = "0.0.0.0", 11434
TARGET_HOST, TARGET_PORT = "host.docker.internal", 11434


async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        writer.close()


async def handle(client_reader, client_writer):
    try:
        target_reader, target_writer = await asyncio.open_connection(
            TARGET_HOST, TARGET_PORT)
    except OSError as e:
        print(f"(ollama_proxy: could not reach {TARGET_HOST}:{TARGET_PORT}: {e})")
        client_writer.close()
        return
    await asyncio.gather(
        pipe(client_reader, target_writer),
        pipe(target_reader, client_writer),
    )


async def main():
    server = await asyncio.start_server(handle, LISTEN_HOST, LISTEN_PORT)
    print(f"[ollama_proxy] forwarding :{LISTEN_PORT} -> {TARGET_HOST}:{TARGET_PORT}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
