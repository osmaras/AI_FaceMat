"""
Assimilate Scratch REST API wrapper.

Provides a reusable ScratchAPI class for interacting with Assimilate Scratch
via the official assimilate_client SDK. All API endpoints are documented
in docs/Assimilate-API-docs-examples/README.md.

Usage:
    from scratch_api import ScratchAPI

    scratch = ScratchAPI()
    selection, shot_data = scratch.get_selected_shot()
"""

import os

import assimilate_client
from assimilate_client import Configuration, ApiClient


class ScratchAPI:
    """
    Wrapper around the Assimilate Scratch REST API.
    Uses ProjectsApi and ApplicationApi for all operations.
    Docs base: http://localhost:8080/APIV2
    """

    def __init__(self, host="127.0.0.1", port=8080):
        config = Configuration()
        config.host = f"http://{host}:{port}/APIV2"
        self.api_client = ApiClient(config)
        self.projects = assimilate_client.ProjectsApi(self.api_client)
        self.app = assimilate_client.ApplicationApi(self.api_client)

    # ---- SHOT SELECTION & PROPERTIES ----

    def get_selected_shot(self):
        """
        Returns (SelectedShotsDataSelection, ShotData) for the first selected
        shot in the current construct, or (None, None) if nothing is selected.
        API: GET /constructs/current/sel_shots?level=ALL
        """
        sel_data = self.projects.get_construct_current_selected_shots(level="ALL")
        if not sel_data or not sel_data.selection:
            return None, None
        sel = sel_data.selection[0]
        shot = sel.shot if sel.shot else self.projects.get_shot(sel.uuid, level="ALL")
        return sel, shot

    def get_shot(self, shot_uuid):
        """Full shot properties. API: GET /shot/{shot_uuid}?level=ALL"""
        return self.projects.get_shot(shot_uuid, level="ALL")

    def get_slot(self, slot_idx):
        """Slot properties (timeline length). API: GET /constructs/current/slots/{slot_idx}"""
        return self.projects.get_construct_current_slot(slot_idx, level="ALL")

    def get_project_paths(self):
        """Current project path configuration. API: GET /projects/current"""
        proj = self.projects.get_projects_current()
        return proj.project_paths if proj and proj.project_paths else None

    # ---- RENDERING (Output Node + Render Queue) ----

    def create_output_node(self, name, output_path, file_format="png",
                           frame_in=None, frame_out=None):
        """
        Create a temporary output node on the current construct, configured
        to render a frame range to the given path.
        API: POST /constructs/current/outputs/new
        Body: ShotData { name, output: { outputpath, extention }, handles: { frame_in, frame_out } }
        Returns the created ShotData (with .uuid).
        """
        shot = assimilate_client.ShotData()
        shot.name = name
        shot.output = assimilate_client.ShotDataOutput()
        shot.output.outputpath = output_path
        shot.output.extention = file_format
        if frame_in is not None and frame_out is not None:
            shot.handles = assimilate_client.ShotDataHandles()
            shot.handles.frame_in = frame_in
            shot.handles.frame_out = frame_out
        return self.projects.add_construct_current_output(shot, level="ALL")

    def start_output_render(self, output_uuid):
        """
        Add an output node to the render queue and start rendering immediately.
        API: POST /application/render/{output_uuid}
        Returns the RenderQueueItem.
        """
        return self.app.add_application_render_queue_item_start(output_uuid)

    def poll_output_render(self, output_uuid):
        """
        Poll a render queue item for progress.
        API: GET /application/render/{output_uuid}
        Returns RenderQueueItem with .status, .frames_done, .frames_total
        """
        return self.app.get_application_render_queue_item(output_uuid)

    def delete_output_node(self, output_uuid):
        """
        Remove a temporary output node from the current construct.
        Rendered files on disk are preserved.
        API: DELETE /constructs/current/outputs/{output_uuid}
        """
        return self.projects.delete_construct_current_output(output_uuid)

    # ---- SNAPSHOT RENDERING (fallback) ----

    def render_frame_snapshot(self, shot_uuid, frame_number, output_path):
        """
        Render a single frame of a shot directly to disk via the snapshot tool.
        API: POST /application/tools/image
        Body: ImageSnapshot { uuid, frame, file, proxy }
        """
        snap = assimilate_client.ImageSnapshot()
        snap.uuid = shot_uuid
        snap.frame = frame_number
        snap.file = output_path
        snap.proxy = False
        return self.app.do_application_render_snapshot(snap)

    # ---- LAYERS ----

    def add_matte_layer(self, shot_uuid, layer_name, matte_dir, slip=0,
                        source_props=None):
        """
        Import a matte frame sequence into Scratch as a layer on a shot.
        1. Create a new shot from the first frame of the matte sequence.
        2. Conform source metadata (timecode, fps, reel_id) onto the matte shot.
        3. Link that shot as a matte layer on the target shot.

        API: POST /shot/new                              (create matte shot)
             PUT  /shot/{matte_shot_uuid}                 (conform metadata)
             POST /shot/{shot_uuid}/layers/new            (create layer)
             PUT  /shot/{shot_uuid}/layers/{layer_idx}/matte  (set matte source)
        """
        # Find the first image file in the matte directory
        matte_files = sorted(f for f in os.listdir(matte_dir)
                             if f.lower().endswith((".png", ".jpg", ".tif", ".tiff")))
        if not matte_files:
            raise FileNotFoundError(f"No image files found in matte directory: {matte_dir}")

        first_frame = os.path.join(matte_dir, matte_files[0])

        # Step 1: Create a shot from the matte sequence
        matte_shot = assimilate_client.ShotData()
        matte_shot.name = layer_name
        matte_shot.file = first_frame
        created = self.projects.add_shot(matte_shot)
        matte_shot_uuid = str(created.uuid) if created and hasattr(created, "uuid") else None
        if not matte_shot_uuid:
            raise RuntimeError(f"Failed to create matte shot for '{layer_name}'")

        # Step 2: Conform source metadata onto the matte shot
        if source_props:
            meta = assimilate_client.ShotData()
            for k, v in source_props.items():
                if v is not None and hasattr(meta, k):
                    setattr(meta, k, v)
            try:
                self.projects.set_shot(meta, matte_shot_uuid)
            except Exception as e:
                print(f"  ⚠️ Could not conform metadata on matte shot: {e}")

        # Step 3: Create an empty layer on the target shot
        layer = assimilate_client.LayerData()
        layer.name = layer_name
        layer.active = True
        self.projects.add_shot_layer(layer, shot_uuid, level="ALL")

        # Step 4: Find the newly created layer index
        layers_data = self.projects.get_shot_layers(shot_uuid=shot_uuid, level="ALL")
        layers = list(getattr(layers_data, "layers", []) or [])
        layer_idx = None
        for idx in range(len(layers) - 1, -1, -1):
            if str(getattr(layers[idx], "name", "")).strip() == layer_name:
                layer_idx = idx
                break
        if layer_idx is None:
            layer_idx = max(0, len(layers) - 1)

        # Step 5: Set the matte source on the layer with slip for frame alignment
        matte_data = assimilate_client.MatteData()
        matte_data.shot_uuid = matte_shot_uuid
        matte_data.blend_mode = "Copy"
        matte_data.map = "Projected"
        matte_data.slip = float(slip)
        self.projects.set_shot_layer_matte(matte_data, shot_uuid, layer_idx)

        return created

    # ---- SHOT CREATION ----

    def create_shot(self, name, media_path):
        """
        Create a new shot from an image sequence directory.
        API: POST /shot/new
        Body: ShotData { name, file }
        Returns ShotData of the created shot.
        """
        shot = assimilate_client.ShotData()
        shot.name = name
        shot.file = media_path
        return self.projects.add_shot(shot)

    # ---- METADATA & NOTES ----

    def set_shot_properties(self, shot_uuid, **props):
        """
        Partial update of shot metadata fields.
        API: PUT /shot/{shot_uuid}
        Body: ShotData with fields to update
        """
        body = assimilate_client.ShotData()
        for k, v in props.items():
            if hasattr(body, k):
                setattr(body, k, v)
        return self.projects.set_shot(body, shot_uuid)

    def append_note(self, shot_uuid, text, frame=0):
        """
        Append a note to a shot's notes list (read-modify-write).
        API: GET /shot/{shot_uuid}  then  PUT /shot/{shot_uuid}
        """
        current = self.projects.get_shot(shot_uuid, level="ALL")
        notes = list(current.notes) if current.notes else []
        note = assimilate_client.NoteData()
        note.note = text
        note.frame = frame
        note.status = 0  # Default note color
        notes.append(note)
        body = assimilate_client.ShotData()
        body.notes = notes
        return self.projects.set_shot(body, shot_uuid)
