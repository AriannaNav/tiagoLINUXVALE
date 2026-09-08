# TIAGo Waiter — HRI course project

A TIAGo robot works as a waiter in a simulated bar. It patrols the room, finds an
occupied table, approaches at a socially appropriate distance, greets the customer,
takes an order by voice, walks to the counter, picks the bottle up with its arm,
and brings it back. It reads the customer's mood and age from the camera, refuses
alcohol to a minor, offers a substitute when an item is out of stock, and reorders
its queue when someone says they are in a hurry.

Every behavioural decision — what to serve, whether to refuse, whether a "yes"
accepts an upsell, how urgent the customer is — is taken by the language model
through a constrained JSON schema. There is no rule table and no regex deciding
any of it. Python only loads the menu, builds the prompt, and passes the model's
own decisions through.

## Where things run

The project is split across two machines-in-one, and this matters for every
install command below.

| Side | What runs there | Why |
|---|---|---|
| Container (`hrai-2025:v1`) | Gazebo, SLAM, Nav2, `waiter_robot` nodes, `hri_bridge.py`, the two frame grabbers | needs ROS 2 Humble |
| Host (WSL) | `waiter.py`, the LLM dialogue, Whisper, the webcam, the Gemini calls, the dashboard | needs the microphone, the webcam and internet |

The two sides talk through JSON files in `hri_project_ffa/shared/`, not through
ROS topics. The host writes `command.json`, the bridge answers in `status.json`.
The dashboard runs on the host because Docker Desktop's `--net host` joins the
Docker VM's network namespace, not WSL's, so a port opened inside the container
is not reachable from the browser.

## Prerequisites

- Docker Desktop with the WSL 2 backend, and the `hrai-2025:v1` image built from
  `dockerfiles/`
- [Ollama](https://ollama.com) running on the host with `qwen2.5:7b` pulled
- A Google Gemini API key
- An NVIDIA GPU is optional. Without it Gazebo runs at roughly half real time and
  Nav2 gets noticeably less reliable.

## Install

### Container side

Nothing to install. The course image already provides ROS 2, `cv_bridge`, OpenCV
and NumPy, which is all the container-side nodes import. `prepare_container.sh`
builds the `waiter_robot` package on first start; `start_waiter.sh` calls it for
you.

### Host side

Ubuntu 3.12 marks its Python as externally managed, so the host dependencies go
in a virtualenv. Keep it on the Linux filesystem rather than under `/mnt/c` —
pip is far slower on the Windows mount, and the mixed ownership between the
container's root and your user causes permission errors on the shared files.

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev portaudio19-dev libasound2-plugins espeak ffmpeg
```

```bash
python3 -m venv ~/.venvs/hrai && ~/.venvs/hrai/bin/pip install -r hri_project_ffa/requirements_host.txt
```

The apt packages cover the parts pip cannot provide: PortAudio behind
`sounddevice` for the microphone, the ALSA-to-PulseAudio plugin that carries the
audio out through WSLg, `espeak` for the robot's voice, and `ffmpeg`, which
Whisper shells out to for decoding.

`openai-whisper` pulls in PyTorch. The first install downloads roughly 2 GB.

Ollama is separate from that virtualenv:

```bash
ollama pull qwen2.5:7b
```

The four ONNX models used for face detection, face identity, emotion and age are
committed in `hri_project_ffa/models/` and load through `cv2.dnn`. No download,
no extra package.

## API key

The Gemini key drives the VLM perception: what is in stock on the counter, and
which tables are occupied. Put it in your shell profile, never in a file inside
the repository.

```bash
echo "export GEMINI_API_KEY='your-key-here'" >> ~/.bashrc && source ~/.bashrc
```

Without it the robot still runs, but it cannot see the counter or the tables and
will say so.

## Run

Terminal 1 — the simulation:

```bash
bash start_waiter.sh
```

It brings up the container, builds `waiter_robot`, launches Gazebo with
`waiter_scene.world`, waits for the robot to spawn and the controllers to come
up, then starts SLAM, Nav2, the overhead camera, the frame grabbers, the bridge
and the dashboard. Expect several minutes on CPU. Add `--headless` to skip the
Gazebo window and watch the cameras on the dashboard instead.

When it prints `Ready.`, open **http://localhost:8077/dashboard.html**.

Terminal 2 — the conversation:

```bash
cd hri_project_ffa && ~/.venvs/hrai/bin/python waiter.py
```

The `cd` matters: `waiter.py` imports `config` from the current directory.

To stop everything:

```bash
bash start_waiter.sh --stop
```

## What to try

| Say this | What should happen |
|---|---|
| "a Coke, please" | walks to the counter, grasps the bottle, delivers it to your table |
| ask for wine at `female02`'s table | refuses and points you to reception — the model decides this, not a rule |
| order something not on the counter | the VLM reports it missing and the robot offers a substitute |
| "I'm in a hurry" | Nav2's speed goes up and the service queue is reordered |
| "where's the restroom?" | points the way, but only if it passed the bathroom on patrol |

The last one is worth watching: the robot answers honestly that it does not know
when the place is not yet in its knowledge graph. That is the open-world
assumption in the graph layer, not a scripted reply.

## Layout

```
exchange/
  start_waiter.sh          one-shot launcher, host side
  prepare_container.sh     build + tuning done inside the container
  waiter_scene.world       the bar: tables, counter, bottles, customers
  overhead_cam.sdf         ceiling camera used for table occupancy
  frame_grabber_tiago.py   head camera  -> shared/frame.jpg
  overhead_grabber.py      ceiling camera -> shared/overhead.jpg
  gazebo_models/           models fetched from Fuel (fetch_models.sh)
  gzclient.sh              re-exports GAZEBO_MODEL_PATH when meshes are missing

  waiter_robot/            ROS 2 package
    arm_controller.py      joint trajectories, gripper, torso, Gazebo attach
    grasp_test.py          the grasp-and-serve sequence, navigation via Nav2
    config/nav2_params.yaml, config/slam_params.yaml

  hri_project_ffa/         host side
    waiter.py              the behaviour loop
    config.py, menu.yaml   scene facts and the menu
    core/                  dialogue agent, language, emotion, face identity
    graph/                 scene graph, temporal manager, ontology, VLM
    ros_nodes/hri_bridge.py   runs in the container, executes the commands
    shared/                the JSON bridge files and the camera frames
    dashboard.html, _serve_dashboard.py
```

`hello_booster.py`, `scan_relay.py`, `teleop_cpp/`, `rviz/` and `tiago.rviz` come
with the course repository and are untouched.

## Configuration

All of these are read on the host by `waiter.py` unless noted.

| Variable | Default | Effect |
|---|---|---|
| `GEMINI_API_KEY` | — | VLM perception; unset disables counter and table sensing |
| `WAITER_LLM` | `1` | `0` falls back to the scripted dialogue |
| `WAITER_AGENT_MODEL` | `qwen2.5:7b` | Ollama model for the dialogue agent |
| `WAITER_AGENT_TEMP` | `0.5` | sampling temperature for the agent |
| `WAITER_GRAPH` | `1` | scene graph and knowledge graph |
| `WAITER_EXPLORE` | `1` | initial patrol that populates the graph |
| `WAITER_PRIORITY_REORDER` | `1` | urgency-driven queue reordering |
| `WAITER_SEED_GRAPH` | `1` | pre-seed the graph to trigger the substitution offer |
| `WAITER_VOICE` | `Samantha` | macOS `say` voice; ignored by `espeak` on Linux |
| `WHISPER_MODEL` | `small` | use `base` if transcription is too slow |
| `FACE_MATCH_THRESHOLD` | `0.363` | SFace cosine distance for recognising a returning customer |
| `HRI_RESTOCK` | `1` | container side: teleport bottles back after each run |

## Troubleshooting

**The robot is missing from Gazebo.** `spawn_entity.py` times out after 30 s and
this world has 88 models, so the first attempt can lose the race. `start_waiter.sh`
retries, but if it still fails: `docker exec tiago_waiter cat /tmp/gazebo.log`.

**Meshes are missing or a customer is absent.** Run `gzclient.sh` inside the
container to rebuild `GAZEBO_MODEL_PATH`, and check that
`gazebo_models/fetch_models.sh` succeeded.

**Nav2 refuses the goal or reports no valid path.** `docker exec tiago_waiter tail -50 /tmp/nav2.log`.
Usually SLAM has not published the `map` frame yet — check `/tmp/slam.log`.

**The robot never hears you.** Verify PulseAudio reaches WSL with `pactl info`;
it should report `unix:/mnt/wslg/PulseServer`.

**Permission errors on the shared JSON files.** The container writes as root.
The bridge chmods what it creates, but a file left over from an earlier run as a
different owner has to be deleted by hand.
