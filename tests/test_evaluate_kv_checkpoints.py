import xml.etree.ElementTree as ET
import zipfile

from evaluate_kv_checkpoints import best_checkpoint_for_run, discover_run_directories, write_xlsx


def test_write_xlsx_writes_metrics_sheet(tmp_path):
    output_path = tmp_path / "metrics.xlsx"
    write_xlsx(output_path, [{"checkpoint": "run/checkpoint-1", "token_accuracy": 0.75}])

    with zipfile.ZipFile(output_path) as archive:
        assert "xl/worksheets/sheet1.xml" in archive.namelist()
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    namespace = {"xlsx": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    assert len(sheet.findall(".//xlsx:row", namespace)) == 2


def test_run_directory_selects_recorded_best_checkpoint(tmp_path):
    run_path = tmp_path / "run_1"
    run_path.mkdir()
    (run_path / "config.json").write_text('{"cli_args": {"metric_for_best_model": "exact_match"}}')
    for step in (10, 20):
        checkpoint = run_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").touch()
    (run_path / "trainer_state.json").write_text(
        '{"best_model_checkpoint": "checkpoint-20", "best_metric": 0.8}'
    )

    assert discover_run_directories(tmp_path) == [run_path]
    checkpoint, metric_name, metric_value = best_checkpoint_for_run(run_path)
    assert checkpoint == run_path / "checkpoint-20" / "model.safetensors"
    assert metric_name == "exact_match"
    assert metric_value == 0.8
