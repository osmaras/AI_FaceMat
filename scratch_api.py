"""
Assimilate Scratch REST API wrapper.

Provides a reusable ScratchAPI class for interacting with Assimilate Scratch
via the official assimilate_client SDK. All API endpoints are documented
in docs/Assimilate-API-docs-examples/README.md.

Usage:
    from scratch_api import ScratchAPI

    scratch = ScratchAPI()
    selection, shot_data = scratch.get_selected_shot()
    render_node = scratch.create_render_shot("My Render", "renders\\#sname.#frame[6].#ext", shot_data.uuid)
    item = scratch.start_render(render_node.uuid)
    item = scratch.poll_render(render_node.uuid)  # call in a loop until item.status != "processing"
"""

import os

import assimilate_client
from assimilate_client import Configuration, ApiClient
from assimilate_client.rest import ApiException


class ScratchAPI:
    """
    Wrapper around the Assimilate Scratch REST API.
    Uses ProjectsApi and ApplicationApi for all operations.
    Docs base: http://localhost:8080/APIV2
    """

    def __init__(self, host="127.0.0.1", port=None, api_key=None):
        port = port or int(os.environ.get("SCRATCH_PORT", "8080"))
        api_key = api_key or os.environ.get("SCRATCH_API_KEY", "")
        config = Configuration()
        config.host = f"http://{host}:{port}/APIV2"
        if api_key:
            config.api_key["Authorization"] = f"Bearer {api_key}"
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

    # ---- RENDERING (Render Shot + Render Queue) ----
    # Tiff/PNG render shot type UUID (constant in Scratch)
    RENDER_SHOT_TYPE_UUID = "00000000-0000-0000-0000-000000000004"

    def create_render_shot(self, name, filespec, input_shot_uuid,
                           file_format="tif", components=4, fmt=0,
                           colorspace="Rec709", eotf="Gamma 2.4",
                           type_uuid=None, outputpath=None):
        """
        Create a render shot and wire an existing shot as its input.
        Follows the RenderSelected pattern: add_shot() + set_shot_input().
        No separate output node required.

        API: POST /shot/new      (create render shot)
             PUT  /shot/{render_uuid}/inputs/0  (set input)

        Args:
            name:            Display name for the render shot.
            filespec:        Output filename pattern, e.g. "#sname.#frame[6].#ext"
                             or "#frame[5].#ext". Keep this to the filename only;
                             pass the directory via `outputpath`.
            input_shot_uuid: UUID of the shot to render.
            file_format:     File extension (default "tif").
            components:      Number of colour components (default 4).
            fmt:             ShotDataOutput format flag (0 = default).
            colorspace:      Output colour space (default "Rec709").
            eotf:            Output EOTF (default "Gamma 2.4").
            type_uuid:       Override the render shot type UUID.
            outputpath:      Absolute directory for rendered files. When set,
                             this is stored in ShotDataOutput.outputpath and
                             `filespec` should contain only the filename pattern.
                             Scratch does not support absolute paths inside
                             `filespec`, so the directory must be split out.

        Returns the created ShotData (with .uuid).
        """
        output_kwargs = dict(
            components=components,
            extention=file_format,
            filespec=filespec,
            format=fmt,
        )
        if outputpath:
            output_kwargs["outputpath"] = outputpath

        sd = assimilate_client.ShotData(
            type_uuid=type_uuid or self.RENDER_SHOT_TYPE_UUID,
            name=name,
            output=assimilate_client.ShotDataOutput(**output_kwargs),
            color_format=assimilate_client.ShotDataColorFormat(
                colorspace=colorspace,
                eotf=eotf,
            ),
        )
        render_node = self.projects.add_shot(sd)

        inp = assimilate_client.InputData(create_copy=True, input_uuid=input_shot_uuid)
        self.projects.set_shot_input(inp, render_node.uuid, 0)

        return render_node

    def start_render(self, render_uuid, delete_existing_media=True):
        """
        Add a render shot to the render queue and start rendering immediately.
        API: POST /application/render/{render_uuid}
        Returns the RenderQueueItem.

        Args:
            render_uuid:           UUID of the render shot to queue.
            delete_existing_media: When True (default), Scratch purges any
                previously rendered media for this render shot before starting.
                Pass False to keep cached frames on disk so the pipeline can
                skip the render pass on subsequent runs.
        """
        dmd = assimilate_client.DeleteMediaData(delete_existing_media=delete_existing_media)
        return self.app.add_application_render_queue_item_start(render_uuid, body=dmd)

    def poll_render(self, render_uuid):
        """
        Poll a render queue item for progress.
        API: GET /application/render/{render_uuid}
        Returns RenderQueueItem with .status, .frames_done, .frames_total
        """
        return self.app.get_application_render_queue_item(render_uuid)

    def delete_render_shot(self, render_uuid, quiet=True):
        """
        Remove a render shot from the project (rendered files on disk are preserved).
        API: DELETE /shot/{render_uuid}

        A render shot that is still in the render queue (or in an errored/locked
        state) cannot be deleted and Scratch returns HTTP 409 Conflict. When
        `quiet` is True (default), such failures are logged and swallowed so
        that pipeline cleanup never masks the original error. Pass
        `quiet=False` to re-raise the ApiException.
        """
        try:
            return self.projects.delete_shot(render_uuid)
        except ApiException as e:
            msg = f"⚠️ Could not delete render shot {render_uuid}: {e}"
            if quiet:
                print(msg)
                return None
            print(msg)
            raise

    # ---- LAYERS ----

    def add_matte_layer(self, shot_uuid, layer_name, matte_dir, slip=0,
                        source_props=None):
        """
        Import a matte frame sequence into Scratch as a layer on a shot.
        1. Create a new shot from the first frame of the matte sequence.
        2. Inject source metadata (timecode, fps, reel_id) onto the matte shot.
        3. Link that shot as a matte layer on the target shot.

        API: POST /shot/new                              (create matte shot)
             PUT  /shot/{matte_shot_uuid}                 (inject metadata)
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

        # Step 2: Inject source metadata onto the matte shot
        if source_props:
            meta = assimilate_client.ShotData()
            for k, v in source_props.items():
                if v is not None and hasattr(meta, k):
                    setattr(meta, k, v)
            try:
                self.projects.set_shot(meta, matte_shot_uuid)
            except Exception as e:
                print(f"  ⚠️ Could not inject metadata on matte shot: {e}")

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
