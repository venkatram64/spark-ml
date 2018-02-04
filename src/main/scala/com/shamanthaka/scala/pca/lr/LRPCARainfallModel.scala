package com.shamanthaka.scala.pca.lr

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Row, SparkSession}

/**
  * Created by Shamanthaka on 12/25/2017.
  */
object LRPCARainfallModel extends App{

  val sparkSession = SparkSession
    .builder()
    .master("local")
    .appName("LRPCARainfallModel")
    .getOrCreate()


  val data = sparkSession.read.format("libsvm").load("weather_libsvm_data.txt")
  //show schema
  data.printSchema()

  val colnames = data.columns
  val firstrow = data.head(1)(0)
  println("\n")
  println("Example Data Row")

  for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
  }

  // Index labels, adding metadata to the label column.
  // Fit on whole dataset to include all labels in index.
  val labelIndexer = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel")
    .fit(data)
  // Automatically identify categorical features, and index them.
  // Set maxCategories so features with > 4 distinct values are treated as continuous.
  val pca = new PCA()
    .setInputCol("features")
    .setOutputCol("pcaFeatures")
    .setK(10)   //10 principal components are chosen
    .fit(data)

  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  // Train a LogisticRegression model.

  val lr = new LogisticRegression()
    .setMaxIter(10)
    .setLabelCol("indexedLabel")
    .setFeaturesCol("pcaFeatures")

  // Convert indexed labels back to original labels.
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  // Chain indexers and forest in a Pipeline.
  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, pca, lr, labelConverter))

  // Train model. This also runs the indexers.
  val model = pipeline.fit(trainingData)

  model.write.overwrite().save("lrPCARAINFALLModel")

  val predictions = model.transform(testData)

  predictions.printSchema()

  // Select example rows to display.
  //predictions.select("prediction", "label", "pcaFeatures").show(100)

  import sparkSession.implicits._
  predictions.select("prediction","label","probability","pcaFeatures")
    .collect()
    .foreach{case Row(prediction: Double, label: Double, probability: Vector, pcaFeatures: Vector) =>
      println(s"($pcaFeatures, $label) -> prob = $probability, prediction=$prediction")
    }

  // Select (prediction, true label) and compute test error.
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  val accuracy = evaluator.evaluate(predictions)
  println("Test Accuracy = " + accuracy * 100)
  println("Test Error = " + (1.0 - accuracy))

/*  val rfModel = model.stages(2).asInstanceOf[LogisticRegressionModel]
  println("Learned classification forest model:\n" + rfModel.)*/

  sparkSession.stop()

}
