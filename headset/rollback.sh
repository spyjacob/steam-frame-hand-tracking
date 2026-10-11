#!/bin/sh
# Undo arcturus-handtrack modifications (see CHANGES.log)
~/handtrack/src/probe/vrsetting camera enableCamera -
echo "$(date -Is) rollback: camera/enableCamera key removed" >> ~/handtrack/CHANGES.log
