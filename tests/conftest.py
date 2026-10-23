# Live-game scripts (tests/live/*) need a running game window and must not
# run in offline CI; collect_ignore_glob keeps pytest from importing them.
collect_ignore_glob = ["live/*"]
