package benchmark

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"

	"lab1/processor"
)

var DefaultWorkers = []int{1, 2, 4, 6, 8, 10, 12, 16, 32, 64, 128}
var DefaultSteps = []int{1, 6}

type RunResult struct {
	StepN    int
	Workers  int
	Duration time.Duration
}

func RunMatrix(
	inputPath string,
	outputDir string,
	threshold float64,
	preset string,
	workerValues []int,
	stepValues []int,
	onRunDone func(RunResult),
) ([]RunResult, error) {
	if outputDir == "" {
		outputDir = "bench_out"
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return nil, err
	}

	results := make([]RunResult, 0, len(workerValues)*len(stepValues))

	for _, stepN := range stepValues {
		for _, workers := range workerValues {
			outputPath := filepath.Join(outputDir, fmt.Sprintf("bench_n%d_w%d.mp4", stepN, workers))

			res, err := processor.ProcessVideo(processor.Config{
				InputPath:   inputPath,
				OutputPath:  outputPath,
				StepN:       stepN,
				Workers:     workers,
				Threshold:   threshold,
				Preset:      preset,
				WriteOutput: true,
			})
			if err != nil {
				return nil, fmt.Errorf("benchmark failed (n=%d, w=%d): %w", stepN, workers, err)
			}

			run := RunResult{
				StepN:    stepN,
				Workers:  workers,
				Duration: res.Duration,
			}
			results = append(results, run)
			if onRunDone != nil {
				onRunDone(run)
			}
		}
	}

	return results, nil
}

func SaveCSV(path string, runs []RunResult) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	if err := w.Write([]string{"n", "workers", "seconds", "duration"}); err != nil {
		return err
	}
	for _, run := range runs {
		row := []string{
			strconv.Itoa(run.StepN),
			strconv.Itoa(run.Workers),
			fmt.Sprintf("%.6f", run.Duration.Seconds()),
			run.Duration.String(),
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}
	return w.Error()
}

func SaveCombinedPlot(path string, runs []RunResult, stepValues []int) error {
	if len(stepValues) == 0 {
		return fmt.Errorf("stepValues is empty")
	}

	p := plot.New()
	p.Title.Text = "Time vs Workers (different n)"
	p.X.Label.Text = "Workers"
	p.Y.Label.Text = "Time (seconds)"
	p.Add(plotter.NewGrid())

	workerSet := make(map[int]struct{})
	for _, run := range runs {
		workerSet[run.Workers] = struct{}{}
	}
	workers := make([]int, 0, len(workerSet))
	for w := range workerSet {
		workers = append(workers, w)
	}
	sort.Ints(workers)

	ticks := make([]plot.Tick, 0, len(workers))
	for _, w := range workers {
		ticks = append(ticks, plot.Tick{
			Value: float64(w),
			Label: strconv.Itoa(w),
		})
	}
	p.X.Tick.Marker = plot.ConstantTicks(ticks)

	palette := []color.RGBA{
		{R: 228, G: 26, B: 28, A: 255},
		{R: 55, G: 126, B: 184, A: 255},
		{R: 77, G: 175, B: 74, A: 255},
		{R: 255, G: 127, B: 0, A: 255},
	}

	for i, stepN := range stepValues {
		filtered := make([]RunResult, 0, len(runs))
		for _, run := range runs {
			if run.StepN == stepN {
				filtered = append(filtered, run)
			}
		}
		if len(filtered) == 0 {
			return fmt.Errorf("no results for n=%d", stepN)
		}

		sort.Slice(filtered, func(i, j int) bool {
			return filtered[i].Workers < filtered[j].Workers
		})

		pts := make(plotter.XYs, len(filtered))
		for j := range filtered {
			pts[j].X = float64(filtered[j].Workers)
			pts[j].Y = filtered[j].Duration.Seconds()
		}

		line, points, err := plotter.NewLinePoints(pts)
		if err != nil {
			return err
		}
		c := palette[i%len(palette)]
		line.Color = c
		points.Color = c
		p.Add(line, points)
		p.Legend.Add(fmt.Sprintf("n=%d", stepN), line, points)
	}
	p.Legend.Top = true
	p.Legend.Left = true

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return p.Save(10*vg.Inch, 4*vg.Inch, path)
}
