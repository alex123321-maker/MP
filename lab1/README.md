# Lab 1: CPU Video Processing

Build:

```powershell
go build -o vidproc.exe .
```

Single run:

```powershell
.\vidproc.exe -in assets\a\clip_20s_30s.mp4 -out assets\a\result.mp4 -n 6 -w 4 -T 25 -preset veryfast
```

Benchmark run:

```powershell
.\vidproc.exe -bench -in assets\a\clip_20s_30s.mp4 -bench-dir assets\a -T 25 -preset veryfast
```

Input videos should be placed under `assets/`. Generated media, profiles and benchmark outputs are ignored by git.
