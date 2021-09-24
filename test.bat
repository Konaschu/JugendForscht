@echo off
set /p ort="Ort: "
start chrome https://www.google.de/maps/place/%ort%/data=!3m1!1e3
timeout 3
snippingtool