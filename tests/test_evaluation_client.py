"""CPU test of the UniVTAC client against an in-process fake policy server (no simulator, no GPU)."""
import threading
import zlib

import numpy as np
import pytest

pytest.importorskip("websockets")
pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
from websockets.sync.server import serve  # noqa: E402

from tacinno.evaluation import msgpack_numpy  # noqa: E402
from tacinno.evaluation.univtac_client import (  # noqa: E402
    InfrastructureError, Policy, TacInnoVLAPolicy, TactileOff, pack_tactile)

H, OFFSETS, HORIZON = 8, [8, 4, 0], 32


class FakeServer:
    """Announces metadata, records every request and answers with a deterministic chunk."""

    def __init__(self, task="lift_can", checkpoint="/ckpt/run/ema_100"):
        self.meta = dict(task=task, checkpoint=checkpoint, proprio_hist=H, use_field=True, tac_hist_offsets=OFFSETS,
                         action_horizon=HORIZON)
        self.requests = []
        self.server = serve(self._handler, "127.0.0.1", 0, compression=None, max_size=None)
        self.port = self.server.socket.getsockname()[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @staticmethod
    def chunk(k):
        c = np.zeros((HORIZON, 8), np.float32)
        c[:, :7] = 0.01 * np.arange(HORIZON)[:, None]  # arm offsets grow along the chunk
        c[:, 7] = k                                    # gripper column = inference index
        return c

    def _handler(self, ws):
        packer = msgpack_numpy.Packer()
        ws.send(packer.pack(self.meta))
        for msg in ws:
            req = msgpack_numpy.unpackb(msg)
            self.requests.append(req)
            ws.send(packer.pack({"model8": self.chunk(int(req["k"]))}))

    def close(self):
        self.server.shutdown()


class FakeTask:
    class cfg:  # noqa: N801
        seed = 1000003

    device = "cpu"

    def __init__(self):
        self.actions = []

    def take_action(self, action, action_type):
        assert action_type == "qpos"
        self.actions.append(action.numpy().copy())


def _obs(step):
    rng = np.random.default_rng(step)
    tac = {name: {"depth": rng.random((240, 320), dtype=np.float32), "marker": rng.random((63, 2), dtype=np.float32)}
           for name in ("left_tactile", "right_tactile")}
    return {"observation": {"head": {"rgb": rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)},
                            "wrist": {"rgb": torch.rand(240, 320, 3)}},
            "embodiment": {"joint": torch.full((9,), float(step))},
            "tactile": tac}


@pytest.fixture()
def server():
    s = FakeServer()
    yield s
    s.close()


def test_chunk_execution_and_request_layout(server):
    policy = Policy(dict(task_name="lift_can", host="127.0.0.1", port=server.port, replan=10,
                         expect_checkpoint="run/ema_100"))
    task = FakeTask()
    for step in range(25):
        policy.eval(task, _obs(step))
    policy.close()

    # queried at steps 0, 10, 20
    assert [int(r["step"]) for r in server.requests] == [0, 10, 20]
    assert [int(r["k"]) for r in server.requests] == [0, 1, 2]
    assert all(int(r["seed"]) == 1000003 for r in server.requests)
    r = server.requests[2]
    assert r["head"].shape == (224, 224, 3) and r["head"].dtype == np.uint8
    assert r["wrist"].shape == (224, 224, 3) and r["wrist"].dtype == np.uint8
    assert r["qhist"].shape == (H, 8) and np.allclose(r["qhist"][:, 0], np.arange(13, 21))
    assert server.requests[0]["qhist"].shape == (1, 8)  # start of the episode: only the steps that exist
    # tactile frames of steps t-8, t-4, t (t = 20), losslessly packed
    assert len(r["tac_hist"]) == 3
    for packet, step in zip(r["tac_hist"], (12, 16, 20)):
        ref = pack_tactile(_obs(step)["tactile"])
        depth = np.frombuffer(zlib.decompress(packet["l_depth_z"]), np.float32).reshape(packet["l_depth_shape"])
        assert np.array_equal(depth, _obs(step)["tactile"]["left_tactile"]["depth"])
        assert np.array_equal(packet["r_marker"], ref["r_marker"])
    # at the episode start the history is clipped to the first frame
    first = server.requests[0]["tac_hist"]
    assert first[0]["l_depth_z"] == first[2]["l_depth_z"]

    # actions: row 1 + (step - query step) of the chunk, arm resolved against the joints of the query step
    acts = np.stack(task.actions)
    assert acts.shape == (25, 8)
    for step in (0, 9, 10, 24):
        s0 = (step // 10) * 10
        row = 1 + step - s0
        assert np.allclose(acts[step, :7], s0 + 0.01 * row)
        assert acts[step, 7] == step // 10


def test_tactile_off_sends_a_fixed_frame(server):
    policy = Policy(dict(task_name="lift_can", host="127.0.0.1", port=server.port, replan=10, tactile_off=True))
    assert isinstance(policy, TactileOff)
    task = FakeTask()
    for step in range(21):
        policy.eval(task, _obs(step))
    policy.reset()
    policy.eval(task, _obs(50))
    policy.close()
    frames = [p["l_depth_z"] for r in server.requests[:3] for p in r["tac_hist"]]
    assert len(set(frames)) == 1  # every packet of the episode is its first frame
    assert server.requests[3]["tac_hist"][0]["l_depth_z"] != frames[0]  # a new episode freezes its own first frame


def test_metadata_checks(server):
    with pytest.raises(InfrastructureError):
        TacInnoVLAPolicy(dict(task_name="insert_hole", host="127.0.0.1", port=server.port))
    with pytest.raises(InfrastructureError):
        TacInnoVLAPolicy(dict(task_name="lift_can", host="127.0.0.1", port=server.port,
                              expect_checkpoint="other/ema_1"))


def test_wire_format_matches_openpi_client():
    openpi_msgpack = pytest.importorskip("openpi_client.msgpack_numpy")
    msg = {"qpos": np.arange(8, dtype=np.float32), "k": 3, "x": np.float32(1.5), "b": b"\x00\x01",
           "nested": [{"a": np.ones((2, 3), np.int64)}]}
    assert msgpack_numpy.Packer().pack(msg) == openpi_msgpack.Packer().pack(msg)
