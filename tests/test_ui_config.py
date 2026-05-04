from yaum.ui import app


def test_start_training_refuses_config_changed_after_prepare(monkeypatch):
    prepared = app.DEFAULT_CONFIG.copy()
    current = prepared.copy()
    current["batch_size"] = prepared["batch_size"] + 8

    monkeypatch.setitem(app.APP_STATE, "model_ready", True)
    monkeypatch.setitem(app.APP_STATE, "thread_active", False)
    monkeypatch.setitem(app.APP_STATE, "prepared_config", prepared)
    monkeypatch.setitem(app.APP_STATE, "config", current)

    status = app.start_training_ui()

    assert "Config changed since Prepare" in status
    assert "batch_size" in status
    assert app.APP_STATE.get("training_thread") is None
