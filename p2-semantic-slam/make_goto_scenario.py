"""
Turn a natural-language question into a Nav2 goal, as a recordable scenario.

This is the link the project is named for. `semantic_map.py` answers "where did
you last see the chair" with a coordinate; this converts that answer into a pose
the robot can be sent to, and writes it out as a scenario the standard recorder
can run. Nothing here is hardcoded to a particular object: change the question
and the goal moves.

Two conversions matter and both are easy to get silently wrong.

  * The semantic map works in world coordinates, because that is where the
    authored object positions live. Nav2 goals are in the map frame, which
    slam_toolbox anchored to the robot's spawn pose. So the goal is transformed
    world -> map on the way out.
  * You cannot drive *to* an object, only *near* it. The goal is set back
    STANDOFF_M along the line from the room centre, facing the object, so the
    robot ends up looking at the thing it was asked about instead of trying to
    occupy the same space.

    python make_goto_scenario.py --query "where did you last see the chair"
"""

import argparse
import json
import math
from pathlib import Path

import semantic_map as sm

HERE = Path(__file__).resolve().parent
STANDOFF_M = 1.0     # stop this far short, so the object is in view not underfoot
WORLD = "/home/sathv/Hanuman/physical-ai/harness/recorder/worlds/tb3_semantic.sdf.xacro"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--query", default="where did you last see the chair")
    p.add_argument("--results",
                   default=str(HERE / "results/map_room.name_object_bearing.llava-phi3.jsonl"))
    p.add_argument("--manifest", default=str(HERE / "data/map_room/manifest.jsonl"))
    p.add_argument("--map-scenario", default=str(HERE / "scenarios/map_room.json"))
    p.add_argument("--map", default=str(HERE / "results/room_map.yaml"))
    p.add_argument("--out", default=str(HERE / "scenarios/goto_query.json"))
    a = p.parse_args()

    obs = sm.load_observations(Path(a.results), Path(a.manifest), poses="slam",
                               use_side=False, scenario=Path(a.map_scenario))
    est, _ = sm.estimate_corroborated(obs)
    if not est:
        est, rng, _ = sm.best_naive(obs)
        print(f"(corroboration located nothing; falling back to naive at {rng} m)")

    target = next((l for l in sorted(est, key=len, reverse=True)
                   if l in a.query.lower()), None)
    if target is None:
        raise SystemExit(f'nothing in the semantic map matches "{a.query}". '
                         f'Known: {", ".join(sorted(est))}')

    ox, oy = est[target]["pos"]
    truth = sm.TRUE_OBJECTS.get(target)
    err = math.hypot(ox - truth[0], oy - truth[1]) if truth else None

    # Approach from the room centre and stop short, facing the object.
    d = math.hypot(ox, oy) or 1.0
    ux, uy = ox / d, oy / d
    gx, gy = ox - STANDOFF_M * ux, oy - STANDOFF_M * uy
    gyaw = math.atan2(oy - gy, ox - gx)

    # world -> map, fixed by where the robot started when the map was built
    spawn = json.loads(Path(a.map_scenario).read_text())["spawn"]
    sx, sy, syaw = spawn.get("x", 0.0), spawn.get("y", 0.0), spawn.get("yaw", 0.0)
    c, s = math.cos(-syaw), math.sin(-syaw)
    mx = c * (gx - sx) - s * (gy - sy)
    my = s * (gx - sx) + c * (gy - sy)
    myaw = gyaw - syaw

    scenario = {
        "id": f"goto_{target}",
        "label": f'"{a.query}" -> drive to the {target}',
        "expect": "",
        # slam_toolbox alone, not Nav2. The full Nav2 stack is fifteen
        # non-composed processes and it drove this four-core box to load 21
        # alongside the simulator, starving the driver before anything moved.
        "nav": False,
        "slam": True,
        "world": WORLD,
        "spawn": {"x": sx, "y": sy, "yaw": syaw},
        "mode": "goto",
        "goal": {"x": round(mx, 3), "y": round(my, 3),
                 "yaw_deg": round(math.degrees(myaw), 1)},
        "timeout_s": 240,
        "duration_cap_s": 1800,
        "_query": a.query,
        "_target": target,
        "_object_world": [round(ox, 3), round(oy, 3)],
        "_object_error_m": round(err, 3) if err is not None else None,
    }
    Path(a.out).write_text(json.dumps(scenario, indent=2) + "\n")

    print(f'query   : "{a.query}"')
    print(f"target  : {target} believed at world ({ox:+.2f}, {oy:+.2f})"
          + (f", {err:.2f} m from truth" if err is not None else ""))
    print(f"goal    : world ({gx:+.2f}, {gy:+.2f}) facing {math.degrees(gyaw):+.0f} deg")
    print(f"          map   ({mx:+.2f}, {my:+.2f}) facing {math.degrees(myaw):+.0f} deg")
    print(f"scenario-> {a.out}")


if __name__ == "__main__":
    main()
