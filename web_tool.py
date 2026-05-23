"""
Bokeh web tool for on-the-fly single-pair WSI registration.

Run with:
    bokeh serve --show web_tool.py
"""

from __future__ import annotations

import base64
import binascii
import traceback
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CheckboxGroup,
    Div,
    FileInput,
    PreText,
    Select,
    TextInput,
)

from batch_run import SUPPORTED_WSI_EXTENSIONS, _run_pair, _validate_wsi_path, _wsi_stem


def _parse_float(value: str, label: str) -> float:
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid number.") from exc


def _parse_int(value: str, label: str) -> int:
    try:
        parsed = int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be > 0.")
    return parsed


title = Div(text="<h2>CORE — On-the-fly WSI Registration</h2>")
launch_info = Div(
    text=(
        "<p>Launch with: <code>bokeh serve --show web_tool.py</code></p>"
        "<p>Open link: <a href='http://localhost:5006/web_tool' target='_blank'>"
        "http://localhost:5006/web_tool</a></p>"
    )
)

source_input = TextInput(title="Source WSI path (moving)", value="")
target_input = TextInput(title="Target WSI path (fixed)", value="")
_accepted_extensions = ",".join(SUPPORTED_WSI_EXTENSIONS)
source_upload = FileInput(title="Or load source WSI file", accept=_accepted_extensions)
target_upload = FileInput(title="Or load target WSI file", accept=_accepted_extensions)
output_dir_input = TextInput(title="Output directory", value="./web_results")

source_mag_input = TextInput(title="Source magnification", value="0.625")
target_mag_input = TextInput(title="Target magnification", value="40.0")
tile_size_input = TextInput(title="OME-TIFF tile size", value="512")
compression_select = Select(
    title="Compression",
    value="lzw",
    options=["lzw", "deflate", "jpeg", "none"],
)
preview_toggle = CheckboxGroup(labels=["Generate checkerboard preview"], active=[0])

run_button = Button(label="Run registration", button_type="success")
status = PreText(text="Ready.")
preview = Div(text="")


def _set_error(message: str, exc: Exception | None = None) -> None:
    details = f"\n\n{traceback.format_exc()}" if exc is not None else ""
    status.text = f"ERROR: {message}{details}"


def _encode_image(image_path: Path) -> str:
    content = image_path.read_bytes()
    encoded = base64.b64encode(content).decode("utf-8")
    return encoded


def _resolve_input_path(
    label: str,
    path_value: str,
    upload_value: str | None,
    upload_filename: str | None,
    output_dir: Path,
) -> str:
    if upload_value and upload_filename:
        upload_dir = output_dir / "_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_name = Path(upload_filename).name
        upload_target = upload_dir / (
            f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S%f')}_{uuid4().hex[:8]}_{label}_{upload_name}"
        )
        try:
            upload_target.write_bytes(base64.b64decode(upload_value))
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Failed to decode uploaded {label} file content.") from exc
        return str(upload_target)
    return path_value.strip()


def _run_registration() -> None:
    output_dir = Path(output_dir_input.value.strip() or "./web_results")
    preview.text = ""

    try:
        source_path = _resolve_input_path(
            label="source",
            path_value=source_input.value,
            upload_value=source_upload.value,
            upload_filename=source_upload.filename,
            output_dir=output_dir,
        )
        target_path = _resolve_input_path(
            label="target",
            path_value=target_input.value,
            upload_value=target_upload.value,
            upload_filename=target_upload.filename,
            output_dir=output_dir,
        )

        if not source_path or not target_path:
            raise ValueError(
                "Please provide both source and target WSI paths or upload both files using the upload buttons."
            )

        source_input.value = source_path
        target_input.value = target_path

        if not Path(source_path).exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        if not Path(target_path).exists():
            raise FileNotFoundError(f"Target path not found: {target_path}")

        _validate_wsi_path(source_path, "source")
        _validate_wsi_path(target_path, "target")

        source_mag = _parse_float(source_mag_input.value, "Source magnification")
        target_mag = _parse_float(target_mag_input.value, "Target magnification")
        tile_size = _parse_int(tile_size_input.value, "OME-TIFF tile size")
        compression = compression_select.value
        visualise = 0 in preview_toggle.active

        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S%f")
        run_id = uuid4().hex[:8]
        stem = _wsi_stem(source_path)
        deformation_output = output_dir / f"{stem}_{stamp}_{run_id}_deformation_field.mha"
        wsi_output = output_dir / f"{stem}_{stamp}_{run_id}_registered.ome.tiff"

        status.text = "Running registration. This can take several minutes..."
        _run_pair(
            source_path=source_path,
            target_path=target_path,
            deformation_output=str(deformation_output),
            wsi_output=str(wsi_output),
            source_magnification=source_mag,
            target_magnification=target_mag,
            tile_size=tile_size,
            compression=compression,
            run_fine_registration=False,
            fixed_nuclei_csv=None,
            moving_nuclei_csv=None,
            visualise=visualise,
        )

        status.text = (
            "Completed.\n"
            f"Deformation field: {deformation_output}\n"
            f"Registered WSI: {wsi_output}"
        )

        if visualise:
            overlay_base = wsi_output.with_suffix("")
            overlay_path = overlay_base.parent / f"{overlay_base.name}_overlay.png"
            if overlay_path.exists():
                encoded_png = _encode_image(overlay_path)
                preview.text = (
                    "<h3>Checkerboard Preview</h3>"
                    f"<img src='data:image/png;base64,{encoded_png}' "
                    "style='max-width: 100%; border: 1px solid #ddd;'/>"
                )
            else:
                preview.text = "<p>Preview image was not generated.</p>"

    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        _set_error(str(exc), exc)


run_button.on_click(_run_registration)

layout = column(
    title,
    launch_info,
    source_input,
    source_upload,
    target_input,
    target_upload,
    output_dir_input,
    row(source_mag_input, target_mag_input, tile_size_input),
    row(compression_select, preview_toggle),
    run_button,
    status,
    preview,
    sizing_mode="stretch_width",
)

curdoc().title = "CORE WSI Registration"
curdoc().add_root(layout)
