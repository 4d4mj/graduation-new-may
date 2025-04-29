import pytest
import httpx

@pytest.mark.integration
def test_full_orchestrator_roundtrip(docker_services):
    # wait until web is healthy
    docker_services.wait_for_service("web", 8000)

    client = httpx.Client(base_url="http://localhost:8000")
    r = client.post("/query", json={
        "query": "What is the weather?",
        "role": "patient",
        "history": []
    })
    assert r.status_code == 200
    data = r.json()
    # agent_name should be either "rag" or "web_search"
    assert data["agent_name"] in {"rag", "web_search"}
    assert "output" in data and isinstance(data["output"]["content"], str)
