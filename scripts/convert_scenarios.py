#!/usr/bin/env python3
"""Offline conversion: JSON scenarios -> binary mmap format.

Converts a directory of Nocturne JSON scenario files into a binary
``.nbin`` format that can be memory-mapped by all worker processes,
so the OS page cache is shared and the first reset of each file is a
mmap instead of a JSON parse.

The binary format is a simple flat layout:

    [header: magic(4) | version(4) | num_objects(4) | num_road_pts(4)]
    [objects: num_objects * sizeof(ObjectRecord)]
    [road_points: num_road_pts * 2 * sizeof(float32)]

where ObjectRecord is a packed struct of:
    id(i8) | x(f4) | y(f4) | length(f4) | width(f4) | heading(f4) | speed(f4) |
    target_x(f4) | target_y(f4) | target_heading(f4) | target_speed(f4) |
    type(i4) | can_block_sight(i1) | can_be_collided(i1) | check_collision(i1)

The conversion is optional: BaseEnv still reads JSON natively. This script
pre-processes scenarios for the mmap fast path.

Usage:
    python scripts/convert_scenarios.py --input /path/to/json_scenarios \
        --output /path/to/binary_scenarios
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


MAGIC = b'NCTN'
VERSION = 1

# ObjectRecord: id(i8) x(f4) y(f4) length(f4) width(f4) heading(f4) speed(f4)
#              target_x(f4) target_y(f4) target_heading(f4) target_speed(f4)
#              type(i4) can_block_sight(i1) can_be_collided(i1) check_collision(i1)
OBJECT_STRUCT = struct.Struct('<q 10f i 3?')
OBJECT_SIZE = OBJECT_STRUCT.size


def _convert_file(json_path: str, out_path: str) -> tuple:
    with open(json_path) as f:
        data = json.load(f)

    objects = data.get('objects', [])
    records = []
    for obj in objects:
        pos = obj.get('position', [{}])
        tgt = obj.get('target_position', [{}])
        pos0 = pos[0] if pos else {}
        tgt0 = tgt[0] if tgt else {}
        obj_type = obj.get('type', 'vehicle')
        type_id = {'vehicle': 0, 'pedestrian': 1, 'cyclist': 2}.get(obj_type, 3)
        records.append(OBJECT_STRUCT.pack(
            int(obj.get('id', 0)),
            float(pos0.get('x', 0.0)), float(pos0.get('y', 0.0)),
            float(obj.get('length', 0.0)), float(obj.get('width', 0.0)),
            float(obj.get('heading', [0.0])[0] if obj.get('heading') else 0.0),
            float(obj.get('speed', [0.0])[0] if obj.get('speed') else 0.0),
            float(tgt0.get('x', 0.0)), float(tgt0.get('y', 0.0)),
            float(obj.get('target_heading', 0.0)),
            float(obj.get('target_speed', 0.0)),
            type_id,
            bool(obj.get('can_block_sight', True)),
            bool(obj.get('can_be_collided', True)),
            bool(obj.get('check_collision', True)),
        ))

    # Road edges
    road_pts = []
    for road in data.get('lanes', []):
        if road.get('road_type') == 'road_edge':
            for pt in road.get('geometry_points', []):
                road_pts.extend([float(pt.get('x', 0.0)), float(pt.get('y', 0.0))])

    num_objects = len(records)
    num_road_pts = len(road_pts) // 2
    header = struct.pack('<4s I I I', MAGIC, VERSION, num_objects, num_road_pts)

    with open(out_path, 'wb') as f:
        f.write(header)
        for rec in records:
            f.write(rec)
        if road_pts:
            f.write(struct.pack(f'<{len(road_pts)}f', *road_pts))

    return num_objects, num_road_pts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True,
                        help='Directory of JSON scenario files.')
    parser.add_argument('--output', required=True,
                        help='Output directory for .nbin files.')
    parser.add_argument('--max-files', type=int, default=-1,
                        help='Max files to convert (-1 = all).')
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_dir.glob('*.json'))
    if args.max_files > 0:
        json_files = json_files[:args.max_files]

    print(f'Converting {len(json_files)} files from {in_dir} -> {out_dir}')
    total_objects = 0
    total_road_pts = 0
    for i, jf in enumerate(json_files):
        out_path = out_dir / (jf.stem + '.nbin')
        try:
            n_obj, n_road = _convert_file(str(jf), str(out_path))
            total_objects += n_obj
            total_road_pts += n_road
        except Exception as exc:
            print(f'  FAILED {jf.name}: {exc}', file=sys.stderr)
            continue
        if (i + 1) % 1000 == 0:
            print(f'  {i + 1}/{len(json_files)} files converted...')

    print(f'Done. {len(json_files)} files, {total_objects} objects, '
          f'{total_road_pts} road points.')


if __name__ == '__main__':
    main()
