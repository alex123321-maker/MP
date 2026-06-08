from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.clustering import BisectingKMeans, GaussianMixture, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, PCA, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


CATEGORICAL = ["segment", "region", "channel"]
NUMERIC = ["age", "income", "tenure_months", "monthly_spend", "support_tickets", "web_visits"]


def generate_dataset(path: Path, rows: int = 3000, seed: int = 42) -> None:
    if path.exists():
        return
    rng = random.Random(seed)
    profiles = [
        ("young_digital", 26, 42000, 14, 95, 1, 34, "digital", "north", "online"),
        ("family_value", 39, 68000, 38, 155, 3, 18, "family", "south", "branch"),
        ("premium_loyal", 51, 112000, 82, 310, 1, 24, "premium", "west", "advisor"),
        ("at_risk", 33, 36000, 8, 65, 7, 9, "basic", "east", "phone"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["customer_id", *NUMERIC, *CATEGORICAL])
        for i in range(rows):
            profile = profiles[rng.randrange(len(profiles))]
            name, age, income, tenure, spend, tickets, visits, segment, region, channel = profile
            writer.writerow([
                i + 1,
                max(18, round(rng.gauss(age, 5))),
                max(18000, round(rng.gauss(income, income * 0.14), 2)),
                max(1, round(rng.gauss(tenure, 10))),
                max(10, round(rng.gauss(spend, spend * 0.18), 2)),
                max(0, round(rng.gauss(tickets, 1.5))),
                max(0, round(rng.gauss(visits, 6))),
                segment if rng.random() > 0.05 else rng.choice([p[7] for p in profiles]),
                region if rng.random() > 0.10 else rng.choice(["north", "south", "east", "west"]),
                channel if rng.random() > 0.08 else rng.choice(["online", "branch", "phone", "advisor"]),
            ])


def build_preprocess_pipeline() -> Pipeline:
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in CATEGORICAL
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{col}_idx" for col in CATEGORICAL],
        outputCols=[f"{col}_oh" for col in CATEGORICAL],
    )
    numeric_assembler = VectorAssembler(inputCols=NUMERIC, outputCol="numeric_vec")
    scaler = StandardScaler(inputCol="numeric_vec", outputCol="numeric_scaled", withMean=True, withStd=True)
    final_assembler = VectorAssembler(
        inputCols=["numeric_scaled", *[f"{col}_oh" for col in CATEGORICAL]],
        outputCol="features",
    )
    return Pipeline(stages=[*indexers, encoder, numeric_assembler, scaler, final_assembler])


def choose_clusters(df, out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction", metricName="silhouette")
    metrics: dict[str, list[dict[str, float]]] = {"kmeans": [], "bisecting": [], "gmm": []}
    best_models: dict[str, Any] = {}
    best_scores: dict[str, tuple[int, float]] = {}
    for k in range(2, 9):
        kmeans = KMeans(k=k, seed=7, maxIter=30, featuresCol="features").fit(df)
        pred = kmeans.transform(df)
        sil = evaluator.evaluate(pred)
        metrics["kmeans"].append({"k": k, "silhouette": sil, "wssse": kmeans.summary.trainingCost})
        if "kmeans" not in best_scores or sil > best_scores["kmeans"][1]:
            best_scores["kmeans"] = (k, sil)
            best_models["kmeans"] = kmeans

        bis = BisectingKMeans(k=k, seed=7, minDivisibleClusterSize=1.0, featuresCol="features").fit(df)
        pred = bis.transform(df)
        sil = evaluator.evaluate(pred)
        metrics["bisecting"].append({"k": k, "silhouette": sil, "wssse": bis.summary.trainingCost})
        if "bisecting" not in best_scores or sil > best_scores["bisecting"][1]:
            best_scores["bisecting"] = (k, sil)
            best_models["bisecting"] = bis

        gmm = GaussianMixture(k=k, seed=7, maxIter=30, featuresCol="features").fit(df)
        pred = gmm.transform(df)
        sil = evaluator.evaluate(pred)
        ll = gmm.summary.logLikelihood
        metrics["gmm"].append({"k": k, "silhouette": sil, "log_likelihood": ll})
        if "gmm" not in best_scores or sil > best_scores["gmm"][1]:
            best_scores["gmm"] = (k, sil)
            best_models["gmm"] = gmm

    draw_metric_plot(out_dir / "cluster_metrics.png", metrics)
    return metrics, {"scores": best_scores, "models": best_models}


def mode_by_cluster(df, cluster_col: str, col: str):
    counts = df.groupBy(cluster_col, col).count()
    max_counts = counts.groupBy(cluster_col).agg(F.max("count").alias("max_count"))
    return (
        counts.join(max_counts, [cluster_col])
        .where(F.col("count") == F.col("max_count"))
        .select(cluster_col, F.col(col).alias(f"{col}_mode"))
    )


def interpret_clusters(df, cluster_col: str = "cluster") -> list[dict[str, Any]]:
    means = df.groupBy(cluster_col).agg(
        F.count("*").alias("size"),
        *[F.round(F.avg(col), 2).alias(f"avg_{col}") for col in NUMERIC],
    )
    result = means
    for col in CATEGORICAL:
        result = result.join(mode_by_cluster(df, cluster_col, col), [cluster_col], "left")
    rows = [row.asDict() for row in result.orderBy(cluster_col).collect()]
    for row in rows:
        income = row["avg_income"]
        spend = row["avg_monthly_spend"]
        tickets = row["avg_support_tickets"]
        if income > 90000 and spend > 220:
            label = "premium high-income customers"
        elif tickets > 4:
            label = "at-risk customers with high support load"
        elif row["avg_age"] < 32:
            label = "young digital customers"
        else:
            label = "mid-market stable customers"
        row["description"] = label
    return rows


def train_classifiers(df, out_dir: Path) -> tuple[list[dict[str, Any]], list[list[int]]]:
    largest_cluster = (
        df.groupBy("cluster")
        .count()
        .orderBy(F.desc("count"))
        .first()["cluster"]
    )
    binary_df = df.withColumn("label", (F.col("cluster") == F.lit(largest_cluster)).cast("double"))
    train, test = binary_df.randomSplit([0.7, 0.3], seed=11)
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

    specs = [
        ("LogisticRegression", LogisticRegression(labelCol="label", featuresCol="features", maxIter=60), "regParam", [0.0, 0.05]),
        ("RandomForest", RandomForestClassifier(labelCol="label", featuresCol="features", seed=5), "numTrees", [20, 50]),
        ("GBT", GBTClassifier(labelCol="label", featuresCol="features", seed=5, maxIter=25), "maxDepth", [3, 5]),
    ]
    rows: list[dict[str, Any]] = []
    best = None
    best_pred = None
    for name, estimator, param_name, values in specs:
        grid = ParamGridBuilder()
        for value in values:
            grid = grid.addGrid(getattr(estimator, param_name), [value])
        tvs = TrainValidationSplit(
            estimator=estimator,
            estimatorParamMaps=grid.build(),
            evaluator=evaluator_f1,
            trainRatio=0.8,
            seed=17,
        )
        start = time.perf_counter()
        model = tvs.fit(train)
        train_s = time.perf_counter() - start
        pred = model.transform(test)
        row = {
            "model": name,
            "accuracy": round(evaluator_acc.evaluate(pred), 4),
            "weighted_precision": round(evaluator_precision.evaluate(pred), 4),
            "weighted_recall": round(evaluator_recall.evaluate(pred), 4),
            "weighted_f1": round(evaluator_f1.evaluate(pred), 4),
            "train_s": round(train_s, 3),
        }
        rows.append(row)
        if best is None or row["weighted_f1"] > best["weighted_f1"]:
            best = row
            best_pred = pred

    labels = sorted([int(row.label) for row in binary_df.select("label").distinct().collect()])
    matrix = []
    for actual in labels:
        matrix_row = []
        for predicted in labels:
            matrix_row.append(best_pred.where((F.col("label") == actual) & (F.col("prediction") == predicted)).count())
        matrix.append(matrix_row)
    draw_classifier_chart(out_dir / "classifier_metrics.png", rows)
    draw_confusion_matrix(out_dir / "confusion_matrix.png", matrix, labels)
    return rows, matrix


def save_pca_plot(df, out_dir: Path) -> None:
    pca = PCA(k=2, inputCol="features", outputCol="pca").fit(df)
    rows = pca.transform(df).select("cluster", "pca").limit(1200).collect()
    points = [(int(row.cluster), float(row.pca[0]), float(row.pca[1])) for row in rows]
    draw_scatter(out_dir / "pca_clusters.png", points)


def draw_metric_plot(path: Path, metrics: dict[str, list[dict[str, float]]]) -> None:
    width, height = 1100, 620
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    left, top, right, bottom = 80, 50, width - 40, height - 90
    draw.rectangle([left, top, right, bottom], outline="black")
    colors = {"kmeans": (40, 100, 200), "bisecting": (40, 160, 80), "gmm": (220, 110, 40)}
    all_vals = [row["silhouette"] for rows in metrics.values() for row in rows]
    ymin, ymax = min(all_vals) * 0.95, max(all_vals) * 1.05

    def px(k: int) -> float:
        return left + (k - 2) / 6 * (right - left)

    def py(v: float) -> float:
        return bottom - (v - ymin) / (ymax - ymin) * (bottom - top)

    for name, rows in metrics.items():
        pts = [(px(int(row["k"])), py(row["silhouette"])) for row in rows]
        draw.line(pts, fill=colors[name], width=3)
        for x, y in pts:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=colors[name])
        draw.text((right - 160, top + 20 + 18 * list(metrics).index(name)), name, fill=colors[name], font=font)
    for k in range(2, 9):
        draw.text((px(k) - 5, bottom + 15), str(k), fill="black", font=font)
    draw.text((width // 2 - 110, 18), "Silhouette by k", fill="black", font=font)
    draw.text((width // 2 - 10, height - 35), "k", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_scatter(path: Path, points: list[tuple[int, float, float]]) -> None:
    width, height = 900, 620
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 70, 50, width - 40, height - 70
    draw.rectangle([left, top, right, bottom], outline="black")
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    colors = [(40, 100, 200), (220, 110, 40), (40, 160, 80), (160, 70, 180), (220, 60, 80), (80, 170, 190)]

    def px(x: float) -> float:
        return left + (x - min(xs)) / (max(xs) - min(xs)) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - min(ys)) / (max(ys) - min(ys)) * (bottom - top)

    for cluster, x, y in points:
        color = colors[cluster % len(colors)]
        draw.ellipse([px(x) - 3, py(y) - 3, px(x) + 3, py(y) + 3], fill=color)
    draw.text((width // 2 - 120, 18), "PCA projection of clusters", fill="black")
    img.save(path)


def draw_classifier_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 900, 560
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 80, 50, width - 40, height - 90
    draw.rectangle([left, top, right, bottom], outline="black")
    bar_w = (right - left) / (len(rows) * 2)
    for i, row in enumerate(rows):
        x0 = left + 55 + i * 2 * bar_w
        y0 = bottom - row["weighted_f1"] * (bottom - top)
        draw.rectangle([x0, y0, x0 + bar_w, bottom], fill=(40, 120, 200))
        draw.text((x0 - 15, y0 - 18), f"{row['weighted_f1']:.3f}", fill="black")
        draw.text((x0 - 25, bottom + 15), row["model"][:12], fill="black")
    draw.text((width // 2 - 120, 18), "Classifier weighted F1", fill="black")
    img.save(path)


def draw_confusion_matrix(path: Path, matrix: list[list[int]], labels: list[int]) -> None:
    cell = 80
    size = cell * (len(labels) + 1)
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    max_val = max(max(row) for row in matrix) or 1
    for i, actual in enumerate(labels):
        draw.text((5, (i + 1) * cell + 30), str(actual), fill="black")
        draw.text(((i + 1) * cell + 30, 5), str(actual), fill="black")
        for j, predicted in enumerate(labels):
            val = matrix[i][j]
            shade = 255 - int(180 * val / max_val)
            x0, y0 = (j + 1) * cell, (i + 1) * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=(shade, shade, 255), outline="black")
            draw.text((x0 + 28, y0 + 30), str(val), fill="black")
    draw.text((cell, 30), "predicted", fill="black")
    draw.text((5, cell - 25), "actual", fill="black")
    img.save(path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="lab6_results")
    parser.add_argument("--rows", type=int, default=3000)
    args = parser.parse_args()
    out_dir = Path(args.out)
    data_path = out_dir / "customer_profiles.csv"
    generate_dataset(data_path, args.rows)

    spark = (
        SparkSession.builder.appName("Lab6SparkMLlib")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        raw = spark.read.csv(str(data_path), header=True, inferSchema=True).na.drop()
        pipeline_model = build_preprocess_pipeline().fit(raw)
        prepared = pipeline_model.transform(raw).cache()
        metrics, best = choose_clusters(prepared, out_dir)
        cluster_model = best["models"]["kmeans"]
        clustered = cluster_model.transform(prepared).withColumnRenamed("prediction", "cluster").withColumn("label", F.col("cluster").cast("double")).cache()
        cluster_summary = interpret_clusters(clustered)
        save_pca_plot(clustered, out_dir)
        classifier_rows, confusion = train_classifiers(clustered, out_dir)
        payload = {
            "dataset": {"rows": raw.count(), "numeric": NUMERIC, "categorical": CATEGORICAL, "source": "synthetic customer profiles"},
            "cluster_metrics": metrics,
            "best_k": {name: {"k": score[0], "silhouette": score[1]} for name, score in best["scores"].items()},
            "cluster_summary": cluster_summary,
            "classifiers": classifier_rows,
            "confusion_matrix": confusion,
        }
        write_json(out_dir / "results.json", payload)
        print(json.dumps(payload["best_k"], ensure_ascii=False, indent=2))
        print(json.dumps(classifier_rows, ensure_ascii=False, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
