from yaum.ui import app


def test_defaults_enable_substrate_diagnostics():
    assert app.DEFAULT_CONFIG["mass_mode"] == "fisher"
    assert app.DEFAULT_CONFIG["adaptive_dt"] is True
    assert app.DEFAULT_CONFIG["snapshot_interval"] > 0
    assert app.DEFAULT_CONFIG["snapshot_on_shift"] is True
    assert app.DEFAULT_CONFIG["reversibility_check_interval"] > 0


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


def test_cli_does_not_share_publicly_by_default():
    args = app.parse_args([])
    assert args.share is False
    assert args.server_name == "127.0.0.1"
    assert app.parse_args(["--share"]).share is True


def test_ui_builds_with_all_components_wired():
    demo = app.create_ui()
    assert demo is not None


def test_new_dynamics_choices_are_accepted():
    for key, value in (("integrator", "leapfrog"), ("model_type", "hamiltonian")):
        old = app.APP_STATE["config"].get(key)
        try:
            assert app.update_config_value(key, value) == value
        finally:
            app.APP_STATE["config"][key] = old
    assert app.update_config_value("friction", -1.0) == app.DEFAULT_CONFIG["friction"]
