import main


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_scan_rejects_non_image(client):
    response = client.post("/scan", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_scan_rejects_invalid_confidence(client):
    response = client.post(
        "/scan",
        params={"conf": 2},
        files={"file": ("card.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "conf" in response.json()["detail"]


def test_admin_sync_hidden_in_production(client, monkeypatch):
    monkeypatch.setattr(main, "APP_ENV", "production")
    response = client.post("/admin/sync-set", params={"set_code": "woe"}, headers={"X-Api-Key": "change-me"})
    assert response.status_code == 404
