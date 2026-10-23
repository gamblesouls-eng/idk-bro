import re

targets = ['220427', '220754', '221055', '221523', '221557', '221627',
           '222255', '222806', '223056', '223242', '223320', '223455']

lines = open(r'logs/hunter_supervisor.log', encoding='utf-8',
             errors='replace').read().splitlines()

result = {}
cur_id = None
frame_ts = None
for l in lines:
    m = re.search(r'encounter=instance1-(\d{8}_\d{6}-\d+)', l)
    if m and m.group(1) != cur_id:
        cur_id = m.group(1)
        frame_ts = None
        continue
    sv = re.search(r'saved debug image .*instance1_encounter_(\d{8}_\d{6})', l)
    if sv and cur_id:
        frame_ts = sv.group(1)
        continue
    ev = re.search(r'target evaluations: (.+)$', l)
    if ev and cur_id and frame_ts:
        result.setdefault(frame_ts, []).append((cur_id, ev.group(1).strip()))

for t in targets:
    full = '20261022_' + t
    hits = result.get(full, [])
    print(t, '=>', hits if hits else 'frame not linked to an evaluation block')
