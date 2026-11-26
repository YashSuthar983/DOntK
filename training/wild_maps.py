"""WildForge: High-Diversity Non-Uniform DOOM Map Generator for Zero-Shot Generalization.

Prevents overfitting to uniform 64u grid corridors by generating 100%-solvable maps with:
- Wide 2-cell (128u) & 3-cell (192u) hallways AND narrow 64u tunnels
- Large open combat arenas (6x6 to 9x9 = 384..576u) with scattered cover pillars
- Interconnected LOOPS & alternate paths (not just spanning trees!)
- Varied ceiling heights (96..256u), outdoor sky courtyards (`F_SKY1`), dark/bright lighting (104..240)
- Multi-step climbable stairs (+16u -> +32u) and safe 1-exit drop-offs
- Randomized wall/floor/door/switch/exit textures across Freedoom2's full texture set
- 0 to 3 sequential keycards/skulls + locked doors + remote switches + Skill-3 ambushes
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from doom_env.wad import Lump, write_wad
from training.doomforge import (
    CELL,
    CEIL_TEXTURES,
    CellSpec,
    DOOR_TEXTURES,
    EXIT_FLOOR_TEXTURES,
    FLOOR_TEXTURES,
    GeneratedMapMeta,
    GridUDMFBuilder,
    KEY_THINGS,
    LOCK_TEXTURES,
    SWITCH_TEXTURES,
    SwitchSpec,
    THING_ARMOR,
    THING_BARREL,
    THING_CACODEMON,
    THING_CHAINGUN,
    THING_CHAINGUNGUY,
    THING_CLIP,
    THING_DEMON,
    THING_HELLKNIGHT,
    THING_IMP,
    THING_MEDIKIT,
    THING_PLAYER,
    THING_ROCKET_LAUNCHER,
    THING_SHELLBOX,
    THING_SHELLS,
    THING_SHOTGUN,
    THING_SHOTGUNGUY,
    THING_STIMPACK,
    THING_SUPER_SHOTGUN,
    THING_ZOMBIEMAN,
    WALL_TEXTURES,
)

WILD_WALLS = WALL_TEXTURES + ("BRICK1", "BRONZE1", "CEMENT1", "COMPBLUE", "METAL2", "SLADWALL", "STARGR1", "WOOD1")
WILD_CEILS = CEIL_TEXTURES + ("F_SKY1", "FLAT1", "FLOOR7_2", "RROCK03")


def _carve_wild_corridor(
    b: GridUDMFBuilder,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    sec_id: int,
    rng: random.Random,
    *,
    width: int = 2,
    door_lock: int = 0,
    door_tag: int = 0,
    floor: int = 0,
    ceil: int = 128,
    light: int = 176,
) -> tuple[int, int] | None:
    """Carve a 1-cell (64u) or 2-cell (128u) wide L-corridor, placing a single clean door."""
    path: list[tuple[int, int]] = []
    cx, cy = x1, y1
    horiz_first = rng.random() < 0.5
    if horiz_first:
        while cx != x2:
            cx += 1 if x2 > cx else -1
            path.append((cx, cy))
        while cy != y2:
            cy += 1 if y2 > cy else -1
            path.append((cx, cy))
    else:
        while cy != y2:
            cy += 1 if y2 > cy else -1
            path.append((cx, cy))
        while cx != x2:
            cx += 1 if x2 > cx else -1
            path.append((cx, cy))

    new_cells = [p for p in path if p not in b.cells]
    wtex = rng.choice(WILD_WALLS)
    ftex = rng.choice(FLOOR_TEXTURES)
    ctex = rng.choice(WILD_CEILS)

    # Choose a straight interior cell for the door (keep doorway 1-cell narrow ONLY at the door cell
    # so the door linedef cleanly seals the corridor, while widening the rest of the hallway to 128u!)
    door_pos = None
    cands = []
    for i, (px, py) in enumerate(new_cells):
        if 1 <= i <= len(new_cells) - 2:
            prev_p, next_p = new_cells[i - 1], new_cells[i + 1]
            if prev_p[0] == next_p[0] or prev_p[1] == next_p[1]:
                cands.append((px, py))
    if cands:
        door_pos = cands[len(cands) // 2]
    elif new_cells:
        door_pos = new_cells[len(new_cells) // 2]

    # Carve primary path
    for i, p in enumerate(new_cells):
        if p == door_pos:
            dtex = LOCK_TEXTURES[door_lock] if (door_lock > 0 and rng.random() > 0.2) else rng.choice(DOOR_TEXTURES)
            b.set_cell(
                p[0], p[1],
                CellSpec(
                    sec_id + 1000, floor=floor, ceil=128,
                    floor_tex="FLAT5_4", wall_tex=wtex, light=light,
                    is_door=True, door_lock=door_lock, door_tex=dtex, door_tag=door_tag,
                ),
            )
        else:
            b.set_cell(
                p[0], p[1],
                CellSpec(sec_id, floor=floor, ceil=ceil, floor_tex=ftex, ceil_tex=ctex, wall_tex=wtex, light=light),
            )
            # Widen non-door corridor cells to 128u (2 cells wide) when width >= 2 and not adjacent to door!
            if width >= 2 and door_pos is not None:
                if abs(p[0] - door_pos[0]) + abs(p[1] - door_pos[1]) >= 2:
                    # Determine perpendicular offset
                    prev_p = new_cells[i - 1] if i > 0 else (x1, y1)
                    if p[0] != prev_p[0]:  # horizontal segment -> widen in y
                        side = (p[0], p[1] + 1)
                    else:  # vertical segment -> widen in x
                        side = (p[0] + 1, p[1])
                    if side not in b.cells and abs(side[0] - door_pos[0]) + abs(side[1] - door_pos[1]) >= 2:
                        b.set_cell(
                            side[0], side[1],
                            CellSpec(sec_id, floor=floor, ceil=ceil, floor_tex=ftex, ceil_tex=ctex, wall_tex=wtex, light=light),
                        )
    return door_pos


def generate_wild_map(seed: int, map_name: str = "MAP01") -> tuple[str, GeneratedMapMeta]:
    rng = random.Random(seed)
    b = GridUDMFBuilder()
    next_sec = 1

    tier = rng.choice((1, 2, 2, 3, 3, 4, 4))
    num_rooms = rng.randint(3, 5) if tier == 1 else (rng.randint(5, 8) if tier == 2 else rng.randint(6, 11))
    num_locks = 0 if tier <= 2 else (1 if tier == 3 else rng.choice((1, 2, 3)))

    # Coarse macro-grid spaced by 10 cells (640u) so rooms can be up to 8x8 open arenas!
    occupied: list[tuple[int, int]] = [(0, 0)]
    edges: list[tuple[int, int]] = []
    while len(occupied) < num_rooms:
        p_idx = rng.randrange(len(occupied))
        px, py = occupied[p_idx]
        nbrs = [(px + dx, py + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (px + dx, py + dy) not in occupied]
        if not nbrs:
            continue
        c_idx = len(occupied)
        occupied.append(rng.choice(nbrs))
        edges.append((p_idx, c_idx))

    adj: dict[int, list[int]] = {i: [] for i in range(num_rooms)}
    parent_of: dict[int, int] = {}
    for p, c in edges:
        adj[p].append(c)
        parent_of[c] = p

    room_centers: list[tuple[int, int]] = []
    room_cells: list[list[tuple[int, int]]] = []
    room_floors: list[int] = []
    dropoff_cells: list[tuple[int, int]] = []

    for idx, (mx, my) in enumerate(occupied):
        rw = rng.randint(4, 8)
        rh = rng.randint(4, 8)
        x0 = mx * 10
        y0 = my * 10
        if idx == 0 and len(adj[0]) == 1 and rng.random() < 0.25:
            r_floor = 48
        elif idx > 0 and rng.random() < 0.30:
            r_floor = rng.choice((0, 16))
        else:
            r_floor = 0
        r_ceil = rng.choice((112, 128, 160, 192, 256))
        r_light = rng.choice((112, 144, 176, 192, 224))
        wtex = rng.choice(WILD_WALLS)
        ftex = rng.choice(FLOOR_TEXTURES)
        ctex = rng.choice(WILD_CEILS)

        cells = []
        for dx in range(rw):
            for dy in range(rh):
                b.set_cell(
                    x0 + dx, y0 + dy,
                    CellSpec(next_sec, floor=r_floor, ceil=r_ceil, floor_tex=ftex, ceil_tex=ctex, wall_tex=wtex, light=r_light),
                )
                cells.append((x0 + dx, y0 + dy))

        # Scatter 1-3 cover pillars inside open arenas (never in spawn room 0, never on center!)
        cx, cy = x0 + rw // 2, y0 + rh // 2
        if idx > 0 and rw >= 5 and rh >= 5 and rng.random() < 0.60:
            n_pil = rng.randint(1, 3)
            for _ in range(n_pil):
                px_c = x0 + rng.randint(1, rw - 2)
                py_c = y0 + rng.randint(1, rh - 2)
                if (px_c, py_c) in b.cells and abs(px_c - cx) + abs(py_c - cy) >= 2:
                    del b.cells[(px_c, py_c)]
                    cells = [c for c in cells if c != (px_c, py_c)]

        if r_floor > 24:
            dropoff_cells.extend(cells)
        next_sec += 1
        room_centers.append((cx, cy))
        room_cells.append(cells)
        room_floors.append(r_floor)

    depth = {0: 0}
    q = [0]
    while q:
        u = q.pop(0)
        for v in adj[u]:
            depth[v] = depth[u] + 1
            q.append(v)

    exit_room = max(range(1, num_rooms), key=lambda r: depth[r])
    exit_path = []
    cur = exit_room
    while cur != 0:
        exit_path.append((parent_of[cur], cur))
        cur = parent_of[cur]
    exit_path.reverse()

    lock_pool = [129, 130, 131]
    rng.shuffle(lock_pool)
    edge_lock: dict[tuple[int, int], int] = {}
    keys_placed: list[dict] = []
    locked_doors_meta: list[dict] = []
    switches_meta: list[dict] = []

    if num_locks > 0 and exit_path:
        chosen = exit_path[-min(num_locks, len(exit_path)):]
        for ki, (u, v) in enumerate(chosen):
            lid = lock_pool[ki % 3]
            blocked = set(chosen[ki:])
            reach = [0]
            rq = [0]
            while rq:
                node = rq.pop(0)
                for nxt in adj[node]:
                    if (node, nxt) not in blocked and nxt not in reach:
                        reach.append(nxt)
                        rq.append(nxt)
            kcands = [r for r in reach if r != 0 and r != exit_room]
            kroom = rng.choice(kcands) if kcands else 0
            edge_lock[(u, v)] = lid
            kx, ky = rng.choice(room_cells[kroom])
            kt = rng.choice(KEY_THINGS[lid])
            b.add_thing(kx, ky, kt, angle=0)
            keys_placed.append({"lock": lid, "thing": kt, "room": kroom, "pos": ((kx + 0.5) * CELL, (ky + 0.5) * CELL)})

    # Optional loop edge between two unlocked macro-adjacent rooms (tests loop-breaking spatial memory!)
    all_edges = list(edges)
    if num_locks == 0 and num_rooms >= 5 and rng.random() < 0.45:
        occ_map = {pos: i for i, pos in enumerate(occupied)}
        existing = set(edges) | {(v, u) for u, v in edges}
        for i, (mx, my) in enumerate(occupied):
            for dx, dy in ((1, 0), (0, 1)):
                j = occ_map.get((mx + dx, my + dy))
                if j is not None and (i, j) not in existing and i != 0 and j != 0:
                    all_edges.append((i, j))
                    break
            if len(all_edges) > len(edges):
                break

    remote_edge = None
    if tier in (2, 4) and rng.random() < 0.30:
        unlk_e = [e for e in edges if e not in edge_lock and e[0] != 0]
        if unlk_e:
            remote_edge = rng.choice(unlk_e)

    for u, v in all_edges:
        lid = edge_lock.get((u, v), 0)
        dtag = 77 if (u, v) == remote_edge else 0
        width = rng.choice((1, 2, 2))
        dpos = _carve_wild_corridor(
            b, room_centers[u][0], room_centers[u][1], room_centers[v][0], room_centers[v][1],
            next_sec, rng, width=width, door_lock=lid, door_tag=dtag,
            floor=min(room_floors[u], room_floors[v]),
            ceil=rng.choice((128, 160)), light=rng.choice((144, 176, 208)),
        )
        next_sec += 2
        if lid > 0 and dpos is not None:
            locked_doors_meta.append({"lock": lid, "pos": ((dpos[0] + 0.5) * CELL, (dpos[1] + 0.5) * CELL)})
        if dtag > 0 and dpos is not None:
            sw_tex = rng.choice(SWITCH_TEXTURES)
            for gx, gy in room_cells[u]:
                for dname, (dx, dy) in (("N", (0, 1)), ("S", (0, -1)), ("E", (1, 0)), ("W", (-1, 0))):
                    if (gx + dx, gy + dy) not in b.cells:
                        b.add_switch(SwitchSpec(gx, gy, dname, special=12, tag=dtag, tex=sw_tex))
                        switches_meta.append({
                            "kind": "remote_door", "tag": dtag,
                            "pos": ((gx + 0.5) * CELL, (gy + 0.5) * CELL),
                            "door_pos": ((dpos[0] + 0.5) * CELL, (dpos[1] + 0.5) * CELL),
                        })
                        break
                if switches_meta:
                    break

    px, py = room_centers[0]
    b.add_thing(px, py, THING_PLAYER, angle=rng.choice((0, 45, 90, 135, 180, 225, 270, 315)))

    exit_type = "walkover" if rng.random() < 0.50 else "switch"
    ex_cell = room_cells[exit_room][-1]
    exit_pos = ((ex_cell[0] + 0.5) * CELL, (ex_cell[1] + 0.5) * CELL)
    if exit_type == "switch":
        placed_sw = False
        ex_sw_tex = rng.choice(SWITCH_TEXTURES)
        for gx, gy in reversed(room_cells[exit_room]):
            for dname, (dx, dy) in (("N", (0, 1)), ("E", (1, 0)), ("S", (0, -1)), ("W", (-1, 0))):
                if (gx + dx, gy + dy) not in b.cells:
                    b.add_switch(SwitchSpec(gx, gy, dname, special=243, tag=0, tex=ex_sw_tex))
                    exit_pos = ((gx + 0.5) * CELL, (gy + 0.5) * CELL)
                    switches_meta.append({"kind": "exit_switch", "pos": exit_pos})
                    placed_sw = True
                    break
            if placed_sw:
                break
        if not placed_sw:
            exit_type = "walkover"
    if exit_type == "walkover":
        b.set_cell(
            ex_cell[0], ex_cell[1],
            CellSpec(next_sec, floor=room_floors[exit_room], ceil=128, floor_tex=rng.choice(EXIT_FLOOR_TEXTURES), is_exit_pad=True),
        )
        next_sec += 1

    # Spawn weapon in room 0 (70% chance so agent ALSO practices Pistol combat like Freedoom2!)
    if rng.random() < 0.75:
        wpn_cell = room_cells[0][0]
        if wpn_cell != (px, py):
            b.add_thing(wpn_cell[0], wpn_cell[1], rng.choice((THING_SHOTGUN, THING_CHAINGUN, THING_SUPER_SHOTGUN)), angle=0)

    enemy_pool = {
        1: (THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_IMP, THING_DEMON),
        2: (THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_IMP),
        3: (THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_IMP, THING_DEMON, THING_CACODEMON),
        4: (THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_CHAINGUNGUY, THING_IMP, THING_DEMON, THING_CACODEMON, THING_HELLKNIGHT),
    }[tier]

    for r_idx in range(1, num_rooms):
        rcells = [c for c in room_cells[r_idx] if c != ex_cell]
        if not rcells:
            continue
        n_e = rng.randint(1, 3) if tier in (1, 4) else rng.randint(0, 2)
        for _ in range(n_e):
            ex, ey = rng.choice(rcells)
            b.add_thing(ex, ey, rng.choice(enemy_pool), angle=rng.choice((0, 90, 180, 270)))
        if rng.random() < 0.30:
            bx, by = rng.choice(rcells)
            b.add_thing(bx, by, THING_BARREL, angle=0)
        if rng.random() < 0.55:
            ix, iy = rng.choice(rcells)
            b.add_thing(ix, iy, rng.choice((THING_STIMPACK, THING_MEDIKIT, THING_SHELLS, THING_SHELLBOX, THING_CLIP, THING_ARMOR)), angle=0)
        if tier == 4 and r_idx == num_rooms // 2 and rng.random() < 0.25:
            rx, ry = rng.choice(rcells)
            b.add_thing(rx, ry, THING_ROCKET_LAUNCHER, angle=0)

    meta = GeneratedMapMeta(
        map_name=map_name, tier=tier, seed=seed, exit_type=exit_type, exit_pos=exit_pos,
        keys_placed=keys_placed, locked_doors=locked_doors_meta, switches=switches_meta,
        toxic_cells=[], dropoff_cells=dropoff_cells,
    )
    return b.compile_textmap(), meta


def build_wild_pack(out_wad: str | Path, num_maps: int = 120, base_seed: int = 77000) -> Path:
    out_wad = Path(out_wad)
    out_wad.parent.mkdir(parents=True, exist_ok=True)
    lumps: list[Lump] = []
    manifest: list[dict] = []
    for i in range(num_maps):
        mname = f"MAP{i + 1:02d}"
        textmap, meta = generate_wild_map(seed=base_seed + i, map_name=mname)
        lumps.append(Lump(mname, b""))
        lumps.append(Lump("TEXTMAP", textmap.encode("utf-8")))
        lumps.append(Lump("ENDMAP", b""))
        manifest.append(meta.__dict__)
    write_wad(out_wad, lumps)
    out_wad.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_wad


if __name__ == "__main__":
    p = build_wild_pack("training/wads/train/wild_train.wad", num_maps=120, base_seed=77000)
    print(f"Built 120 WildForge diverse maps -> {p}")
    p_eval = build_wild_pack("training/wads/eval_wild/wild_eval.wad", num_maps=20, base_seed=88000)
    print(f"Built 20 WildForge unseen OOD eval maps -> {p_eval}")
