package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"time"

	"lab1/benchmark"
	"lab1/processor"
)

func main() {
	inPath := flag.String("in", "", "входной mp4")
	outPath := flag.String("out", "out.mp4", "выходной mp4")
	n := flag.Int("n", 1, "шаг n (1..6)")
	workers := flag.Int("w", 2, "количество потоков для обработки кадров")
	threshold := flag.Float64("T", 30, "порог T для ||diff|| (примерно 0..441)")
	preset := flag.String("preset", "veryfast", "preset для libx264 (ultrafast/superfast/veryfast/...)")

	benchMode := flag.Bool("bench", false, "запустить серию бенчмарков (workers x n)")
	benchDir := flag.String("bench-dir", "benchmarks", "директория для CSV и графиков бенчмарка")

	cpuProf := flag.String("cpuprofile", "", "записать CPU профиль в файл")
	memProf := flag.String("memprofile", "", "записать heap профиль в файл в конце")
	traceOut := flag.String("trace", "", "записать trace в файл")

	flag.Parse()

	if *inPath == "" {
		fmt.Fprintln(os.Stderr, "Ошибка: укажи входной файл через -in input.mp4")
		os.Exit(2)
	}

	var (
		stopTrace func()
		stopCPU   func()
	)
	var err error

	stopTrace, err = startTrace(*traceOut)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Ошибка trace: %v\n", err)
		os.Exit(1)
	}
	defer stopTrace()

	stopCPU, err = startCPUProfile(*cpuProf)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Ошибка cpuprofile: %v\n", err)
		os.Exit(1)
	}
	defer stopCPU()

	startedAt := time.Now()
	if *benchMode {
		if err := runBenchmark(*inPath, *threshold, *preset, *benchDir); err != nil {
			fmt.Fprintf(os.Stderr, "Ошибка benchmark: %v\n", err)
			os.Exit(1)
		}
	} else {
		if err := runSingle(*inPath, *outPath, *n, *workers, *threshold, *preset); err != nil {
			fmt.Fprintf(os.Stderr, "Ошибка обработки: %v\n", err)
			os.Exit(1)
		}
	}

	if err := writeMemProfile(*memProf); err != nil {
		fmt.Fprintf(os.Stderr, "Ошибка memprofile: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("done in %s\n", time.Since(startedAt))
}

func runSingle(inputPath, outputPath string, stepN, workers int, threshold float64, preset string) error {
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
		return err
	}

	fmt.Printf("w=%d h=%d fps=%.3f frames=%d n=%d T=%.2f workers=%d out=%s\n",
		res.Width, res.Height, res.FPS, res.Frames, stepN, threshold, workers, outputPath)
	fmt.Printf("processing time: %s\n", res.Duration)
	return nil
}

func runBenchmark(inputPath string, threshold float64, preset string, benchDir string) error {
	fmt.Printf("benchmark input=%s\n", inputPath)
	fmt.Printf("workers=%v n=%v\n", benchmark.DefaultWorkers, benchmark.DefaultSteps)
	benchOutDir := filepath.Join(benchDir, "video_out")

	runs, err := benchmark.RunMatrix(
		inputPath,
		benchOutDir,
		threshold,
		preset,
		benchmark.DefaultWorkers,
		benchmark.DefaultSteps,
		func(run benchmark.RunResult) {
			fmt.Printf("n=%d w=%d -> %s\n", run.StepN, run.Workers, run.Duration)
		},
	)
	if err != nil {
		return err
	}

	csvPath := filepath.Join(benchDir, "results.csv")
	if err := benchmark.SaveCSV(csvPath, runs); err != nil {
		return err
	}

	plotCombined := filepath.Join(benchDir, "time_vs_workers_n1_n6.png")
	if err := benchmark.SaveCombinedPlot(plotCombined, runs, benchmark.DefaultSteps); err != nil {
		return err
	}

	fmt.Printf("saved: %s\n", csvPath)
	fmt.Printf("saved: %s\n", plotCombined)
	fmt.Printf("saved videos: %s\n", benchOutDir)
	return nil
}

func startTrace(path string) (func(), error) {
	if path == "" {
		return func() {}, nil
	}

	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	if err := trace.Start(f); err != nil {
		_ = f.Close()
		return nil, err
	}
	return func() {
		trace.Stop()
		_ = f.Close()
	}, nil
}

func startCPUProfile(path string) (func(), error) {
	if path == "" {
		return func() {}, nil
	}

	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		_ = f.Close()
		return nil, err
	}
	return func() {
		pprof.StopCPUProfile()
		_ = f.Close()
	}, nil
}

func writeMemProfile(path string) error {
	if path == "" {
		return nil
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	runtime.GC()
	return pprof.WriteHeapProfile(f)
}
