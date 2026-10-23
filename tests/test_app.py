import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import app as agent


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.sqlite3"
    monkeypatch.setattr(agent, "DB_PATH", str(db_path))
    agent.initialize_database()
    with TestClient(agent.app) as tc:
        yield tc


def headers(token="user-one", content_type=True, version="1.0"):
    h = {"Authorization": f"Bearer {token}", "A2A-Version": version}
    if content_type:
        h["Content-Type"] = agent.A2A_MIME
    return h


def sample_batch():
    return {
        "batchId": "BATCH-1",
        "policyRevision": "POLICY-1",
        "packages": [{
            "packageId": "PKG-1",
            "documents": "Reconciled invoice data [INV-01] plus policy authorization [POL-02] and receipt lookup [PAY-03].",
        }],
    }


def sample_decision(package):
    return {
        "packageId": package["packageId"],
        "action": "settle_invoice",
        "facts": {"vendorName": "Example Vendor", "invoiceNumber": "INV-100", "amountMinor": 12345, "currency": "INR"},
        "evidenceRefs": ["[INV-01]", "[POL-02]", "[PAY-03]"],
        "rationale": "The proposed action is settle_invoice because the invoice is reconciled and authorized in [INV-01] and [POL-02], with payment status corroborated by [PAY-03].",
    }


def test_public_card_has_exact_base_url(client, monkeypatch):
    monkeypatch.setenv("A2A_BASE_URL", "https://example.onrender.com/a2a/")
    r = client.get("/.well-known/agent-card.json")
    assert r.status_code == 200
    card = r.json()
    assert card["supportedInterfaces"][0]["url"] == "https://example.onrender.com/a2a/"
    assert agent.BATCH_MIME in card["defaultInputModes"]
    assert set([agent.PROPOSALS_MIME, agent.RECEIPTS_MIME]).issubset(card["defaultOutputModes"])


def test_version_and_auth_required(client):
    r = client.get("/a2a/tasks", headers={"A2A-Version": "1.0"})
    assert r.status_code == 401
    r = client.get("/a2a/tasks", headers={"Authorization": "Bearer x", "A2A-Version": "0.9"})
    assert r.status_code == 400


def test_initial_task_then_accepted_receipt(client, monkeypatch):
    async def mock_call(batch_data, packages):
        return {p["packageId"]: sample_decision(p) for p in packages}
    monkeypatch.setattr(agent, "call_groq", mock_call)
    message = {
        "messageId": "msg-initial-1",
        "role": "ROLE_USER",
        "parts": [{"mediaType": agent.BATCH_MIME, "data": sample_batch()}],
    }
    r = client.post("/a2a/message:send", headers=headers(), content=json.dumps({"message": message, "configuration": {"returnImmediately": False}}))
    assert r.status_code == 200, r.text
    task = r.json()["task"]
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    proposal_data = task["artifacts"][0]["parts"][0]["data"]
    proposal = proposal_data["proposals"][0]
    assert len(proposal["evidenceRefs"]) == 3
    assert len(proposal["actionId"]) >= 12

    continuation = {
        "messageId": "msg-results-1",
        "taskId": task["id"],
        "contextId": task["contextId"],
        "role": "ROLE_USER",
        "parts": [{"mediaType": agent.RESULTS_MIME, "data": {
            "batchId": "BATCH-1",
            "results": [{
                "packageId": proposal["packageId"],
                "actionId": proposal["actionId"],
                "action": proposal["action"],
                "outcome": "ACCEPTED",
                "receiptNonce": "nonce-unique-1",
            }],
        }}],
    }
    r2 = client.post("/a2a/message:send", headers=headers(), content=json.dumps({"message": continuation}))
    assert r2.status_code == 200, r2.text
    completed = r2.json()["task"]
    assert completed["status"]["state"] == "TASK_STATE_COMPLETED"
    receipt_parts = [p for a in completed["artifacts"] for p in a["parts"] if p["mediaType"] == agent.RECEIPTS_MIME]
    assert len(receipt_parts) == 1
    assert receipt_parts[0]["data"]["executions"][0]["receiptNonce"] == "nonce-unique-1"


def test_idempotency_replay_and_conflict(client, monkeypatch):
    calls = {"n": 0}
    async def mock_call(batch_data, packages):
        calls["n"] += 1
        return {p["packageId"]: sample_decision(p) for p in packages}
    monkeypatch.setattr(agent, "call_groq", mock_call)
    message = {"messageId": "same-id", "role": "ROLE_USER", "parts": [{"mediaType": agent.BATCH_MIME, "data": sample_batch()}]}
    one = client.post("/a2a/message:send", headers=headers(), content=json.dumps({"message": message, "configuration": {"returnImmediately": False}}))
    two = client.post("/a2a/message:send", headers=headers(), content=json.dumps({"message": message, "configuration": {"returnImmediately": True}}))
    assert one.status_code == two.status_code == 200
    assert one.json()["task"]["id"] == two.json()["task"]["id"]
    assert calls["n"] == 1
    changed = json.loads(json.dumps(message))
    changed["parts"][0]["data"]["batchId"] = "OTHER-BATCH"
    conflict = client.post("/a2a/message:send", headers=headers(), content=json.dumps({"message": changed}))
    assert conflict.status_code == 409
    assert conflict.json()["error"] == "IDEMPOTENCY_CONFLICT"


def test_user_isolation(client, monkeypatch):
    async def mock_call(batch_data, packages):
        return {p["packageId"]: sample_decision(p) for p in packages}
    monkeypatch.setattr(agent, "call_groq", mock_call)
    message = {"messageId": "owner-msg", "role": "ROLE_USER", "parts": [{"mediaType": agent.BATCH_MIME, "data": sample_batch()}]}
    response = client.post("/a2a/message:send", headers=headers("owner"), content=json.dumps({"message": message}))
    task_id = response.json()["task"]["id"]
    assert client.get(f"/a2a/tasks/{task_id}", headers=headers("outsider", content_type=False)).status_code == 404
    assert client.get("/a2a/tasks", headers=headers("outsider", content_type=False)).json() == {"tasks": []}


def test_cancel_prevents_receipt(client, monkeypatch):
    async def mock_call(batch_data, packages):
        return {p["packageId"]: sample_decision(p) for p in packages}
    monkeypatch.setattr(agent, "call_groq", mock_call)
    message = {"messageId": "cancel-msg", "role": "ROLE_USER", "parts": [{"mediaType": agent.BATCH_MIME, "data": sample_batch()}]}
    response = client.post("/a2a/message:send", headers=headers(), content=json.dumps({"message": message}))
    task = response.json()["task"]
    canceled = client.post(f"/a2a/tasks/{task['id']}:cancel", headers=headers(), content="{}")
    assert canceled.status_code == 200
    assert canceled.json()["status"]["state"] == "TASK_STATE_CANCELED"
    assert not any(p["mediaType"] == agent.RECEIPTS_MIME for a in canceled.json()["artifacts"] for p in a["parts"])
    again = client.post(f"/a2a/tasks/{task['id']}:cancel", headers=headers(), content="{}")
    assert again.status_code == 409
