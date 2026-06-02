import requests


class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def list_sessions(self) -> list[dict]:
        response = requests.get(f"{self.base_url}/sessions", timeout=20)
        response.raise_for_status()
        return response.json()

    def create_session(self, title: str | None = None) -> dict:
        payload = {"title": title} if title else {}
        response = requests.post(f"{self.base_url}/sessions", json=payload, timeout=20)
        response.raise_for_status()
        return response.json()

    def get_messages(self, session_id: str) -> list[dict]:
        response = requests.get(f"{self.base_url}/sessions/{session_id}/messages", timeout=20)
        response.raise_for_status()
        return response.json()

    def stream_chat(self, session_id: str, message: str):
        response = requests.post(
            f"{self.base_url}/sessions/{session_id}/chat",
            json={"message": message},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data:"):
                continue
            # Preserve token spacing. SSE lines are "data: <chunk>" and chunk
            # may intentionally contain leading/trailing whitespace/punctuation.
            chunk = raw[len("data:") :]
            if chunk.startswith(" "):
                chunk = chunk[1:]
            if chunk == "[DONE]":
                break
            yield chunk

    def list_memories(self) -> dict:
        response = requests.get(f"{self.base_url}/memories", timeout=20)
        response.raise_for_status()
        return response.json()

    def update_memory(self, memory_id: str, content: str) -> dict:
        response = requests.patch(
            f"{self.base_url}/memories/{memory_id}",
            json={"content": content},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    def forget_memory(self, memory_id: str) -> None:
        response = requests.delete(f"{self.base_url}/memories/{memory_id}", timeout=20)
        response.raise_for_status()

    def end_session(self, session_id: str, generate_episodic: bool = True) -> None:
        response = requests.post(
            f"{self.base_url}/sessions/{session_id}/end",
            json={"generate_episodic": generate_episodic},
            timeout=20,
        )
        response.raise_for_status()
