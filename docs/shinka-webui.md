# Local Shinka interface

The native ShinkaEvolve interface is available at
[http://127.0.0.1:8000/](http://127.0.0.1:8000/) while the local server runs.
The repository launcher binds only to loopback and exposes these two selected
archives through the native read-only database interface:

| Archive | Programs | Selection score |
| :--- | :--- | :--- |
| `adaptive-shinka-20261003/shinka` | Executable mutation-width rules | Equal-weight active and previous-task return |
| `search-20261002-r1/shinka` | Static mutation width and archive fraction | Active-task return |

From the repository directory, start the server with:

```bash
.venv/bin/python -u scripts/run_shinka_webui.py --port 8000
```

Stop a foreground server with **Ctrl+C**. The background server started during
the interactive experiment records its PID and command in
`results/webui-20261003/server.json` and its log in
`results/webui-20261003/server.log`; stopping that PID leaves the experiments intact.
Other archives inside this repository's `results` directory can be selected
with repeated `--archive PATH` arguments.

Select an archive, then select a node in **Tree** to inspect **Code**, **Diff**,
**Eval**, and **LLM**. **Programs** lists the candidates; **Metrics** shows their
recorded measurements. The tree checks for new programs every three seconds.
Refresh the dashboard after a new archive is first created. An archive appears
after its first program is recorded.

Tree generations are outer Shinka proposal slots, including the initial program
at generation zero. Each evaluated candidate contains its own multi-generation
GA training runs. The two archives use different objectives, so their displayed
scores are not a direct comparison of the algorithms. Native dollar columns
are token-price estimates recorded by Shinka, not subscription billing records.

The launcher serves the installed, pinned Shinka interface without changing its
files. It also forces read-only SQLite connections for the native quick-stat
and failed-proposal helpers. Initial verification caught those helpers opening
write-capable connections and checkpointing the completed static archive's
existing write-ahead log. The database's physical checksum changed; all 25
programs, 20 archive members, event/failure tables, and saved-state artifact
receipts still matched the published evidence. The wrapper was repaired before
opening the adaptive archive; no experiment records were rewritten.

The graph page loads the native external JavaScript dependencies. The
pinned Plotly URL emits a nonfatal deprecation notice in the browser console;
the tree, program inspection, and metric plots remain functional.
