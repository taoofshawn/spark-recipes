#!/usr/bin/env python3
"""CPU checks for RoCEnante on a switchless ring (b12x.comm.roce_ring + scripts/ring_mesh/plan.py).

- B12X_ROCE_PEER_HCA_MAPS: one ';'-separated map per rank is picked by rank, B12X_ROCE_PEER_HCA_MAP
  still wins, a wrong count refuses.
- plan.py orients the ring from the fabric subnets (sparkring numbers it with physical port f0
  clockwise), refuses a cable that breaks that contract, and translates sparkring's per-rank
  paths into peer maps for the TP rank order. The fixtures are a real four-Spark ring whose TP
  order runs the other way round the cables (TP 0,1,2,3 = ring 0,3,2,1).
- --reversed: a node with both plugs swapped is accepted with its roles swapped in software, a
  node with only one plane swapped still refuses.
- every map pairs path i at both ends on the same PCIe domain, and a neighbour's paths cross the
  cable (f0 at one end, f1 at the other).
- in the image: the RoCEnante overlay selects the package from DSV41_ROCE_RING.

No GPU, torch or network. usage: python3 tests/test_roce_ring.py
"""
import ast
import hashlib
import importlib.util
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ROCE_RING = ROOT / "runtime/b12x/b12x/comm/roce_ring"
UINT32_MASK = 0xFFFFFFFF
CTRL = 0x1000
EPOCH = 0x2000


def _load(name, *candidates):
    for c in candidates:
        if c.is_file():
            spec = importlib.util.spec_from_file_location(name, c)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(candidates)


paths = _load("roce_ring_paths", ROOT / "runtime/b12x/b12x/comm/roce_ring/_path_config.py",
              Path("/opt/b12x/b12x/comm/roce_ring/_path_config.py"))
plan = _load("ring_mesh_plan", ROOT / "scripts/ring_mesh/plan.py")


class Uint32(int):
    def __new__(cls, value):
        return super().__new__(cls, int(value) & UINT32_MASK)

    def __add__(self, other):
        return Uint32(int(self) + int(other))


def simulate_timeout(filename, paths, sequence, timeout):
    """Run one kernel's timeout control flow against CPU-backed words."""
    source = ROCE_RING / filename
    tree = ast.parse(source.read_text())
    kernels = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "kernel"
    ]
    if len(kernels) != 1:
        raise AssertionError(f"expected one kernel in {source}, found {len(kernels)}")
    kernel = kernels[0]
    kernel.decorator_list = []
    for argument in kernel.args.args:
        argument.annotation = None
    kernel.returns = None
    memory = {EPOCH: Uint32(sequence - 1)}
    waits = []

    def load(address):
        return Uint32(memory.get(address, 0))

    def store(address, value):
        memory[address] = Uint32(value)

    def atomic_add(address, value):
        previous = load(address)
        store(address, previous + value)
        return previous

    def spin(address, expected, limit):
        waits.append((address, int(expected), int(limit)))
        return Uint32(timeout)

    namespace = {
        "Uint32": Uint32,
        "Int32": int,
        "Int64": int,
        "PACK_BYTES": 16,
        "PATH_COUNT": 2,
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                thread_idx=lambda: (0, 0, 0),
                block_idx=lambda: (0, 0, 0),
                grid_dim=lambda: (1, 1, 1),
                sync_threads=lambda: None,
            )
        ),
        "cutlass": SimpleNamespace(const_expr=bool, range_constexpr=range),
        "ld_relaxed_gpu_u32": load,
        "ld_relaxed_sys_u32": load,
        "st_relaxed_sys_u32": store,
        "st_release_sys_u32": store,
        "st_release_gpu_u32": store,
        "atomic_add_relaxed_gpu_u32": atomic_add,
        "fence_sc_sys": lambda: None,
        "fence_sc_gpu": lambda: None,
        "spin_until_eq_acquire_sys": spin,
    }
    exec(compile(ast.Module(body=[kernel], type_ignores=[]), str(source), "exec"), namespace)
    runtime = SimpleNamespace(
        _threads=32,
        _world_size=4,
        _rank=1,
        _slots=2,
        _opposite_paths=paths,
        _flag_stride=128,
    )
    pointer = SimpleNamespace(toint=lambda: 0x3000)
    arguments = dict(
        self=runtime,
        input_ptr=pointer,
        output_ptr=pointer,
        nbytes=0,
        recv_base=0x4000,
        flag_base=0x8000,
        send_base=0xC000,
        ctrl_base=CTRL,
        slot_bytes=16,
        epoch_ptr=EPOCH,
        spin_limit=Uint32(1),
    )
    if filename == "_allgather_cute.py":
        arguments.update(shard_packs=0, row_packs=1)
    else:
        arguments.update(size_packs=0)
    namespace["kernel"](**arguments)
    if not waits:
        raise AssertionError("the simulated thread did not execute a peer wait")
    if any(expected != sequence for _, expected, _ in waits):
        raise AssertionError(f"kernel waited for a sequence other than {sequence}")
    return memory

HOSTS = ["s1", "s2", "s3", "s4"]  # TP rank order
# netdev -> IPv4 per host; each DAC is its own /24, f0 of every node cabled to the f1 of the node before it in TP order
ADDR = {
    "s1": {"enp1s0f0np0": "10.10.3.11/24", "enp1s0f1np1": "10.10.0.10/24", "enP2p1s0f0np0": "10.20.3.11/24", "enP2p1s0f1np1": "10.20.0.10/24"},
    "s2": {"enp1s0f0np0": "10.10.0.11/24", "enp1s0f1np1": "10.10.1.10/24", "enP2p1s0f0np0": "10.20.0.11/24", "enP2p1s0f1np1": "10.20.1.10/24"},
    "s3": {"enp1s0f0np0": "10.10.1.11/24", "enp1s0f1np1": "10.10.2.10/24", "enP2p1s0f0np0": "10.20.1.11/24", "enP2p1s0f1np1": "10.20.2.10/24"},
    "s4": {"enp1s0f0np0": "10.10.2.11/24", "enp1s0f1np1": "10.10.3.10/24", "enP2p1s0f0np0": "10.20.2.11/24", "enP2p1s0f1np1": "10.20.3.10/24"},
}
# sparkring rocenante_native_path_arguments() for that ring (ring rank -> "dest,function,rdma,gid,hops")
NATIVE = {
    "0": ["1,0,rocep1s0f0,3,1", "1,1,roceP2p1s0f0,3,1", "2,0,rocep1s0f0,3,2", "2,1,roceP2p1s0f1,3,2", "3,0,rocep1s0f1,3,1", "3,1,roceP2p1s0f1,3,1"],
    "1": ["0,0,rocep1s0f1,3,1", "0,1,roceP2p1s0f1,3,1", "2,0,rocep1s0f0,3,1", "2,1,roceP2p1s0f0,3,1", "3,0,rocep1s0f0,3,2", "3,1,roceP2p1s0f1,3,2"],
    "2": ["0,0,rocep1s0f1,3,2", "0,1,roceP2p1s0f0,3,2", "1,0,rocep1s0f1,3,1", "1,1,roceP2p1s0f1,3,1", "3,0,rocep1s0f0,3,1", "3,1,roceP2p1s0f0,3,1"],
    "3": ["0,0,rocep1s0f0,3,1", "0,1,roceP2p1s0f0,3,1", "1,0,rocep1s0f1,3,2", "1,1,roceP2p1s0f0,3,2", "2,0,rocep1s0f1,3,1", "2,1,roceP2p1s0f1,3,1"],
}
MAPS = "1=1/3,2=0/3,3=0/2;0=0/2,2=1/3,3=1/2;0=1/2,1=0/2,3=1/3;0=1/3,1=0/3,2=0/2"


def inventory(addr):
    return {h: {"ports": {nd: {"ipv4": ip} for nd, ip in ports.items()}} for h, ports in addr.items()}


class PeerMapEnv(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.pop(k, None) for k in (paths.PEER_HCA_MAP_ENV, paths.PEER_HCA_MAPS_ENV)}

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_per_rank_selection(self):
        os.environ[paths.PEER_HCA_MAPS_ENV] = MAPS
        self.assertEqual(paths.peer_hca_map(4, 0, 4), ((-1, -1), (1, 3), (0, 3), (0, 2)))
        self.assertEqual(paths.peer_hca_map(4, 3, 4), ((1, 3), (0, 3), (0, 2), (-1, -1)))

    def test_single_map_wins(self):
        os.environ[paths.PEER_HCA_MAPS_ENV] = MAPS
        os.environ[paths.PEER_HCA_MAP_ENV] = "1=0/2,2=0/3,3=1/3"
        self.assertEqual(paths.peer_hca_map(4, 0, 4)[1], (0, 2))

    def test_wrong_count_refuses(self):
        os.environ[paths.PEER_HCA_MAPS_ENV] = MAPS.rsplit(";", 1)[0]
        with self.assertRaises(ValueError):
            paths.peer_hca_map(4, 0, 4)

    def test_two_hca_default_unchanged(self):
        self.assertEqual(paths.peer_hca_map(4, 1, 2), ((0, 1), (-1, -1), (0, 1), (0, 1)))


class Plan(unittest.TestCase):
    def test_ring_order(self):
        self.assertEqual(plan.ring_order(inventory(ADDR), HOSTS), ["s1", "s4", "s3", "s2"])

    def test_miscabled_refuses(self):
        bad = {h: dict(p) for h, p in ADDR.items()}
        bad["s2"]["enp1s0f0np0"], bad["s2"]["enp1s0f1np1"] = bad["s2"]["enp1s0f1np1"], bad["s2"]["enp1s0f0np0"]
        with self.assertRaises(SystemExit):
            plan.ring_order(inventory(bad), HOSTS)

    def test_reversed_node(self):
        swapped = {h: dict(p) for h, p in ADDR.items()}
        for f0, f1 in (("enp1s0f0np0", "enp1s0f1np1"), ("enP2p1s0f0np0", "enP2p1s0f1np1")):
            swapped["s2"][f0], swapped["s2"][f1] = swapped["s2"][f1], swapped["s2"][f0]
        with self.assertRaises(SystemExit):
            plan.ring_order(inventory(swapped), HOSTS)
        self.assertEqual(plan.ring_order(inventory(swapped), HOSTS, {"s2"}), ["s1", "s4", "s3", "s2"])
        self.assertEqual([plan.port("s2", "clockwise", f, {"s2"}) for f in (0, 1)], ["enp1s0f1np1", "enP2p1s0f1np1"])
        half = {h: dict(p) for h, p in ADDR.items()}
        half["s2"]["enp1s0f0np0"], half["s2"]["enp1s0f1np1"] = half["s2"]["enp1s0f1np1"], half["s2"]["enp1s0f0np0"]
        with self.assertRaises(SystemExit):
            plan.ring_order(inventory(half), HOSTS, {"s2"})

    def test_peer_maps(self):
        order = plan.ring_order(inventory(ADDR), HOSTS)
        self.assertEqual(plan.peer_maps(NATIVE, order, HOSTS), MAPS)

    def test_maps_pair_paths_consistently(self):
        m = [dict((int(p), tuple(map(int, v.split("/")))) for p, v in (e.split("=") for e in r.split(","))) for r in MAPS.split(";")]
        domain, port = (lambda h: h // 2), (lambda h: h % 2)
        for r in range(4):
            for p in m[r]:
                for i in (0, 1):
                    self.assertEqual(domain(m[r][p][i]), domain(m[p][r][i]), (r, p, i))
                    if (p - r) % 4 in (1, 3):
                        self.assertNotEqual(port(m[r][p][i]), port(m[p][r][i]), (r, p, i))


class TimeoutContract(unittest.TestCase):
    def test_provenance_hashes_cover_the_current_overlay(self):
        provenance = json.loads((ROOT / "runtime/b12x/roce_ring-provenance.json").read_text())
        self.assertEqual(set(provenance["source_files_sha256"]), {
            path.name for path in ROCE_RING.iterdir() if path.is_file()
        })
        for name, expected in provenance["source_files_sha256"].items():
            actual = hashlib.sha256((ROCE_RING / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)

    def test_timeout_remains_visible_across_uint32_wrap(self):
        for filename in ("_allgather_cute.py", "_oneshot_cute.py"):
            for paths in (2, 4):
                for sequence in (1, 0x80000000, 0xFFFFFFFF, 0):
                    with self.subTest(filename=filename, paths=paths, sequence=sequence):
                        memory = simulate_timeout(filename, paths, sequence, timeout=True)
                        self.assertEqual(memory.get(CTRL + 24, 0), 1)
                        self.assertEqual(memory.get(EPOCH + 12, 0), 1)
                        self.assertEqual(memory[CTRL + 8], sequence)
                        self.assertEqual(memory[EPOCH], (sequence - 1) & UINT32_MASK)

    def test_successful_wrap_advances_epoch(self):
        for filename in ("_allgather_cute.py", "_oneshot_cute.py"):
            for paths in (2, 4):
                with self.subTest(filename=filename, paths=paths):
                    memory = simulate_timeout(filename, paths, 0, timeout=False)
                    self.assertEqual(memory[EPOCH], 0)
                    self.assertEqual(memory.get(CTRL + 24, 0), 0)
                    self.assertEqual(memory.get(EPOCH + 12, 0), 0)

    def test_proxy_diagnostic_counters_are_atomic(self):
        source = (ROCE_RING / "_roce_proxy.c").read_text()
        self.assertIn("atomic_uint_least32_t last_seq;", source)
        self.assertIn("atomic_uint_fast64_t ops_posted;", source)
        self.assertIn("atomic_uint_fast64_t writes_completed;", source)
        self.assertNotIn("\n    uint32_t last_seq;", source)
        self.assertNotIn("\n    uint64_t ops_posted;", source)
        self.assertNotIn("\n    uint64_t writes_completed;", source)

    def test_timeout_flag_has_a_distinct_control_word(self):
        runtime = (ROCE_RING / "roce_oneshot.py").read_text()
        self.assertIn("self._layout.ctrl_off + 28", runtime)
        self.assertIn("int(self._ctrl_np[6]) != 0", runtime)
        self.assertIn('"timed_out": bool(self._ctrl_words[6].item())', runtime)

    def test_host_health_uses_flag_and_unsigned_sequence(self):
        source = ROCE_RING / "roce_oneshot.py"
        tree = ast.parse(source.read_text())
        owner = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RoceOneshotAllReduce"
        )
        selected = []
        for node in owner.body:
            if isinstance(node, ast.FunctionDef) and node.name in {"check_health", "poisoned"}:
                node.decorator_list = []
                selected.append(node)
        namespace = {}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
        for sequence in (0, 1, 0x80000000, 0xFFFFFFFF):
            signed = sequence if sequence < 0x80000000 else sequence - (1 << 32)
            runtime = SimpleNamespace(
                _proxy=None,
                _ctrl_np=[0, 0, signed, 2, 0, 0, 1],
                rank=1,
            )
            with self.subTest(sequence=sequence):
                self.assertTrue(namespace["poisoned"](runtime))
                with self.assertRaisesRegex(RuntimeError, f"at sequence {sequence};"):
                    namespace["check_health"](runtime)
        runtime._ctrl_np = [0] * 7
        self.assertFalse(namespace["poisoned"](runtime))
        namespace["check_health"](runtime)


class ImageOverlay(unittest.TestCase):
    PYNCCL = Path("/sgl-workspace/sglang/python/sglang/srt/distributed/device_communicators/pynccl.py")

    @unittest.skipUnless(PYNCCL.is_file(), "image only")
    def test_overlay_selects_package(self):
        src = self.PYNCCL.read_text()
        self.assertIn('"b12x.comm.roce_ring" if os.environ.get("DSV41_ROCE_RING", "0") == "1" else "b12x.comm.roce"', src)
        self.assertNotIn("from b12x.comm import roce\n", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
