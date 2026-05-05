package processor

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Config struct {
	InputPath    string
	OutputPath   string
	StepN        int
	Workers      int
	Threshold    float64
	Preset       string
	WriteOutput  bool
	ProbeTimeout time.Duration
}

type Result struct {
	Width    int
	Height   int
	FPS      float64
	Frames   int
	Duration time.Duration
}

type ffprobeJSON struct {
	Streams []struct {
		Width      int    `json:"width"`
		Height     int    `json:"height"`
		RFrameRate string `json:"r_frame_rate"`
	} `json:"streams"`
}

type workerPart struct {
	idx []int
	val []int
	top []int
}

type significantJob struct {
	part       *workerPart
	cur, prev  []byte
	start, end int
	T2         int
	wg         *sync.WaitGroup
}

type topJob struct {
	part    *workerPart
	cut     int
	topMask []uint32
	mark    uint32
	wg      *sync.WaitGroup
}

type paintJob struct {
	topSet   []int
	start    int
	end      int
	w        int
	h        int
	topMask  []uint32
	mark     uint32
	outFrame []byte
	wg       *sync.WaitGroup
}

func parseFraction(s string) float64 {
	parts := strings.Split(s, "/")
	if len(parts) != 2 {
		v, _ := strconv.ParseFloat(s, 64)
		return v
	}
	a, _ := strconv.ParseFloat(parts[0], 64)
	b, _ := strconv.ParseFloat(parts[1], 64)
	if b == 0 {
		return 0
	}
	return a / b
}

func ffprobe(path string, timeout time.Duration) (w, h int, fps float64, err error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "ffprobe",
		"-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "stream=width,height,r_frame_rate",
		"-of", "json",
		path,
	)
	out, err := cmd.Output()
	if err != nil {
		return 0, 0, 0, fmt.Errorf("ffprobe: %w", err)
	}

	var j ffprobeJSON
	if err := json.Unmarshal(out, &j); err != nil {
		return 0, 0, 0, err
	}
	if len(j.Streams) == 0 {
		return 0, 0, 0, fmt.Errorf("ffprobe: no video stream")
	}

	w, h = j.Streams[0].Width, j.Streams[0].Height
	fps = parseFraction(j.Streams[0].RFrameRate)
	return w, h, fps, nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func writeFrame(encIn io.WriteCloser, buf []byte) error {
	if encIn == nil {
		return nil
	}
	written := 0
	for written < len(buf) {
		n, err := encIn.Write(buf[written:])
		if err != nil {
			return err
		}
		written += n
	}
	return nil
}

func killRunning(dec, enc *exec.Cmd) {
	if dec != nil && dec.Process != nil {
		_ = dec.Process.Kill()
	}
	if enc != nil && enc.Process != nil {
		_ = enc.Process.Kill()
	}
}

func collectSignificantWorker(
	part *workerPart,
	cur []byte,
	prev []byte,
	start int,
	end int,
	T2 int,
) {
	localIdx := part.idx[:0]
	localVal := part.val[:0]

	for p := start; p < end; p++ {
		off := p * 3
		dr := int(cur[off]) - int(prev[off])
		dg := int(cur[off+1]) - int(prev[off+1])
		db := int(cur[off+2]) - int(prev[off+2])
		d2 := dr*dr + dg*dg + db*db

		if d2 >= T2 {
			localIdx = append(localIdx, p)
			localVal = append(localVal, d2)
		}
	}

	part.idx = localIdx
	part.val = localVal
}

func collectTopWorker(
	part *workerPart,
	cut int,
	topMask []uint32,
	mark uint32,
) {
	localTop := part.top[:0]
	for i := 0; i < len(part.idx); i++ {
		if part.val[i] >= cut {
			p := part.idx[i]
			topMask[p] = mark
			localTop = append(localTop, p)
		}
	}
	part.top = localTop
}

func paintBordersWorker(
	topSet []int,
	start int,
	end int,
	w int,
	h int,
	topMask []uint32,
	mark uint32,
	outFrame []byte,
) {
	for i := start; i < end; i++ {
		p := topSet[i]
		x := p % w
		y := p / w

		isBorder := x == 0 || x == w-1 || y == 0 || y == h-1
		for dy := -1; dy <= 1 && !isBorder; dy++ {
			yy := y + dy
			for dx := -1; dx <= 1; dx++ {
				if dx == 0 && dy == 0 {
					continue
				}
				xx := x + dx
				np := yy*w + xx
				if topMask[np] == mark {
					isBorder = true
					break
				}
			}
		}

		if isBorder {
			off := p * 3
			outFrame[off] = 255
			outFrame[off+1] = 0
			outFrame[off+2] = 0
		}
	}
}

func ProcessVideo(cfg Config) (Result, error) {
	if cfg.InputPath == "" {
		return Result{}, fmt.Errorf("input path is required")
	}
	if cfg.StepN < 1 || cfg.StepN > 6 {
		return Result{}, fmt.Errorf("n must be in [1..6]")
	}
	if cfg.Workers < 1 {
		return Result{}, fmt.Errorf("workers must be >= 1")
	}
	if cfg.WriteOutput && cfg.OutputPath == "" {
		return Result{}, fmt.Errorf("output path is required when WriteOutput=true")
	}
	if cfg.Preset == "" {
		cfg.Preset = "veryfast"
	}
	if cfg.ProbeTimeout <= 0 {
		cfg.ProbeTimeout = 5 * time.Second
	}

	startedAt := time.Now()

	w, h, fps, err := ffprobe(cfg.InputPath, cfg.ProbeTimeout)
	if err != nil {
		return Result{}, err
	}
	if fps <= 0 {
		fps = 30
	}

	frameSize := w * h * 3
	pixelCount := w * h
	T2 := int(cfg.Threshold * cfg.Threshold)
	const maxD2 = 255 * 255 * 3

	dec := exec.Command("ffmpeg",
		"-v", "error",
		"-i", cfg.InputPath,
		"-an",
		"-f", "rawvideo",
		"-pix_fmt", "rgb24",
		"-vsync", "0",
		"pipe:1",
	)
	decOut, err := dec.StdoutPipe()
	if err != nil {
		return Result{}, err
	}

	var enc *exec.Cmd
	var encIn io.WriteCloser
	if cfg.WriteOutput {
		enc = exec.Command("ffmpeg",
			"-y",
			"-v", "error",
			"-f", "rawvideo",
			"-pix_fmt", "rgb24",
			"-s", fmt.Sprintf("%dx%d", w, h),
			"-r", fmt.Sprintf("%.6f", fps),
			"-i", "pipe:0",
			"-an",
			"-c:v", "libx264",
			"-preset", cfg.Preset,
			"-crf", "18",
			cfg.OutputPath,
		)
		encIn, err = enc.StdinPipe()
		if err != nil {
			return Result{}, err
		}
	}

	devNull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		return Result{}, err
	}
	defer devNull.Close()

	dec.Stderr = devNull
	if enc != nil {
		enc.Stderr = devNull
	}

	if err := dec.Start(); err != nil {
		return Result{}, err
	}
	if enc != nil {
		if err := enc.Start(); err != nil {
			killRunning(dec, nil)
			_ = dec.Wait()
			return Result{}, err
		}
	}

	reader := bufio.NewReaderSize(decOut, frameSize*4)
	ringSize := cfg.StepN + 1
	ring := make([][]byte, ringSize)
	for i := 0; i < ringSize; i++ {
		ring[i] = make([]byte, frameSize)
	}

	outFrame := make([]byte, frameSize)
	topMask := make([]uint32, pixelCount)
	hist := make([]int, maxD2+1)
	touched := make([]int, 0, maxD2+1)
	topSet := make([]int, 0, pixelCount/20)

	parts := make([]workerPart, cfg.Workers)
	chunk := (pixelCount + cfg.Workers - 1) / cfg.Workers
	capHint := chunk / 8
	if capHint < 1024 {
		capHint = 1024
	}
	for i := 0; i < cfg.Workers; i++ {
		parts[i].idx = make([]int, 0, capHint)
		parts[i].val = make([]int, 0, capHint)
		parts[i].top = make([]int, 0, capHint/10+1)
	}

	significantJobs := make([]chan significantJob, cfg.Workers)
	topJobs := make([]chan topJob, cfg.Workers)
	paintJobs := make([]chan paintJob, cfg.Workers)
	for i := 0; i < cfg.Workers; i++ {
		significantJobs[i] = make(chan significantJob, 1)
		topJobs[i] = make(chan topJob, 1)
		paintJobs[i] = make(chan paintJob, 1)

		sigCh := significantJobs[i]
		topCh := topJobs[i]
		paintCh := paintJobs[i]

		go func() {
			for job := range sigCh {
				collectSignificantWorker(job.part, job.cur, job.prev, job.start, job.end, job.T2)
				job.wg.Done()
			}
		}()
		go func() {
			for job := range topCh {
				collectTopWorker(job.part, job.cut, job.topMask, job.mark)
				job.wg.Done()
			}
		}()
		go func() {
			for job := range paintCh {
				paintBordersWorker(
					job.topSet, job.start, job.end,
					job.w, job.h, job.topMask, job.mark, job.outFrame,
				)
				job.wg.Done()
			}
		}()
	}
	defer func() {
		for i := 0; i < cfg.Workers; i++ {
			close(significantJobs[i])
			close(topJobs[i])
			close(paintJobs[i])
		}
	}()

	frameIdx := 0
	for {
		cur := ring[frameIdx%ringSize]
		_, err := io.ReadFull(reader, cur)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			killRunning(dec, enc)
			return Result{}, err
		}

		if frameIdx < cfg.StepN {
			if err := writeFrame(encIn, cur); err != nil {
				killRunning(dec, enc)
				return Result{}, err
			}
			frameIdx++
			continue
		}

		prev := ring[(frameIdx-cfg.StepN)%ringSize]
		mark := uint32(frameIdx + 1)

		totalSignificant := 0
		var wg sync.WaitGroup
		wg.Add(cfg.Workers)
		for wid := 0; wid < cfg.Workers; wid++ {
			start := wid * chunk
			end := minInt(start+chunk, pixelCount)
			if start >= pixelCount {
				start = pixelCount
				end = pixelCount
			}
			significantJobs[wid] <- significantJob{
				part:  &parts[wid],
				cur:   cur,
				prev:  prev,
				start: start,
				end:   end,
				T2:    T2,
				wg:    &wg,
			}
		}
		wg.Wait()

		for _, d2 := range touched {
			hist[d2] = 0
		}
		touched = touched[:0]

		for wid := 0; wid < cfg.Workers; wid++ {
			localVals := parts[wid].val
			totalSignificant += len(localVals)
			for i := 0; i < len(localVals); i++ {
				d2 := localVals[i]
				if hist[d2] == 0 {
					touched = append(touched, d2)
				}
				hist[d2]++
			}
		}
		if totalSignificant == 0 {
			if err := writeFrame(encIn, cur); err != nil {
				killRunning(dec, enc)
				return Result{}, err
			}
			frameIdx++
			continue
		}

		k := int(math.Ceil(0.10 * float64(totalSignificant)))
		if k < 1 {
			k = 1
		}
		acc := 0
		cut := 0
		for d2 := maxD2; d2 >= 0; d2-- {
			c := hist[d2]
			if c == 0 {
				continue
			}
			acc += c
			if acc >= k {
				cut = d2
				break
			}
		}

		topSet = topSet[:0]
		wg.Add(cfg.Workers)
		for wid := 0; wid < cfg.Workers; wid++ {
			topJobs[wid] <- topJob{
				part:    &parts[wid],
				cut:     cut,
				topMask: topMask,
				mark:    mark,
				wg:      &wg,
			}
		}
		wg.Wait()

		for wid := 0; wid < cfg.Workers; wid++ {
			topSet = append(topSet, parts[wid].top...)
		}

		if len(topSet) == 0 {
			if err := writeFrame(encIn, cur); err != nil {
				killRunning(dec, enc)
				return Result{}, err
			}
			frameIdx++
			continue
		}

		copy(outFrame, cur)

		topCount := len(topSet)
		base := topCount / cfg.Workers
		rem := topCount % cfg.Workers
		wg.Add(cfg.Workers)
		for wid := 0; wid < cfg.Workers; wid++ {
			start := wid*base + minInt(wid, rem)
			end := start + base
			if wid < rem {
				end++
			}
			paintJobs[wid] <- paintJob{
				topSet:   topSet,
				start:    start,
				end:      end,
				w:        w,
				h:        h,
				topMask:  topMask,
				mark:     mark,
				outFrame: outFrame,
				wg:       &wg,
			}
		}
		wg.Wait()

		if err := writeFrame(encIn, outFrame); err != nil {
			killRunning(dec, enc)
			return Result{}, err
		}
		frameIdx++
	}

	if encIn != nil {
		if err := encIn.Close(); err != nil {
			killRunning(dec, enc)
			return Result{}, err
		}
	}
	if err := dec.Wait(); err != nil {
		killRunning(nil, enc)
		return Result{}, err
	}
	if enc != nil {
		if err := enc.Wait(); err != nil {
			return Result{}, err
		}
	}

	return Result{
		Width:    w,
		Height:   h,
		FPS:      fps,
		Frames:   frameIdx,
		Duration: time.Since(startedAt),
	}, nil
}
