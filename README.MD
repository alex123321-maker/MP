go build -o vidproc.exe .
go tool pprof -http=:880 cpu.pprof

.\vidproc.exe -bench -in assets\a\clip_20s_30s.mp4 -bench-dir assets\a -T 25 -preset veryfast
.\vidproc.exe -in assets\a\clip_20s_30s.mp4 -out assets\a\result.mp4 -n 6 -w 4 -T 25 -preset veryfast
