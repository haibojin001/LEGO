# Frozen paper split

`lego_repo_522.txt` contains the 522 task IDs common to the historical A and
B per-task record sets. It is sorted for stable diffs. Recomputing difficulty
from the frozen task files gives D1–D5 counts of `117/111/116/96/82`, exactly
the paper's inventory. Of the 565 frozen task directories, the other 43 never
entered those records.

The 522-task denominator includes 11 tasks above the current 300-module
generation cap. They retain their membership and receive zero when the
runner returns `too-many-modules`. `benchmark/build_index.py` builds a separate
`lego_repo_available.txt` for tasks within the cap; that split is useful for
development, but it is a different population and must be labelled as such.

The fixed list has 522 unique task IDs. The checked-in inventory CSV provides
task-level metadata without distributing the source trees.

Task directories are distributed separately because they include upstream
sources and tests. `benchmark/build_index.py` checks that every ID is present
and that the frozen band counts still match before a paper-split run.
